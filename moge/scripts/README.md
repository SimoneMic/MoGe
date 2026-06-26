# MoGe HSP scripts

This folder contains the Humanoid Sensing and Perception (HSP) helper scripts
built on top of [MoGe](https://github.com/microsoft/MoGe) monocular geometry
estimation. Two of them are documented here:

- [`depth_runtime.py`](depth_runtime.py) — real-time RGB→depth estimation over
  YARP ports (online use on the robot).
- [`depthize_dataset.py`](depthize_dataset.py) — offline batch processing that
  adds a MoGe-estimated depth stream to a [LeRobot](https://github.com/huggingface/lerobot)
  dataset and pushes the result to the Hugging Face Hub.

Both use the **MoGe-2 ViT-L** checkpoint (`Ruicheng/moge-2-vitl`), which is
downloaded automatically from the Hugging Face Hub on first run, and both
require a CUDA-capable GPU (`device = "cuda"`).

The easiest way to get every dependency (CUDA, YARP with Python bindings,
LeRobot, MoGe) in one place is to use the Docker image — see
[Docker](#docker) at the bottom.

---

## `depth_runtime.py`

Runs MoGe continuously as a YARP module: it receives RGB frames on an input
port and publishes the estimated depth as a grayscale image on an output port.
This is the script meant to run live on the robot.

### Ports

| Port | Direction | Type | Description |
|------|-----------|------|-------------|
| `/moGe/rgb:i`   | input  | `ImageRgb`  | RGB frames to process |
| `/moGe/depth:o` | output | `ImageMono` | Estimated depth, encoded to 8-bit |

The depth output is **not** raw metres. Metric depth is clipped to
`[0, max_depth]` (default `max_depth = 8.0` m) and linearly mapped to `0–255`,
so a pixel value of `255` means "≥ 8 m or invalid". Invalid/masked pixels
(MoGe returns `+inf`) saturate to `255`.

### Decoding back to metres

The encoding is `u8 = clip(depth_m / max_depth, 0, 1) * 255`. To recover the
(clipped) metric depth on the consumer side, invert it with the **same**
`max_depth` used when encoding (`8.0` m by default):

```
depth_m = (u8 / 255) * max_depth
```

```python
import numpy as np

MAX_DEPTH_M = 8.0  # must match the encoder (max_depth / MAX_DEPTH_M)

def decode_depth(u8: np.ndarray, max_depth: float = MAX_DEPTH_M) -> np.ndarray:
    """uint8 depth image -> metric depth in metres (clipped to [0, max_depth])."""
    return u8.astype(np.float32) / 255.0 * max_depth
```

The mapping is lossy: precision is `max_depth / 255 ≈ 0.031 m` per level at the
default 8 m range, and `255` (≥ `max_depth` or invalid) cannot be told apart
from a genuine 8 m reading.

### Requirements

- A running **YARP name server** (`yarpserver`) reachable on the network.
- YARP Python bindings (`import yarp` must work).
- A GPU (the model is loaded onto `cuda`).
- An RGB source publishing on a YARP port (e.g. a camera `grabber` device).

### Run

```bash
# 1. Make sure a YARP server is reachable
yarpserver --write   # (only if one isn't already running)

# 2. Start the module
python moge/scripts/depth_runtime.py

# 3. Connect an RGB source to the input port, e.g. a camera publishing on
#    /cam/rgb:o
yarp connect /cam/rgb:o /moGe/rgb:i

# 4. View / consume the depth output
yarp connect /moGe/depth:o /viewer mjpeg     # e.g. with yarpview
```

Stop it with `Ctrl-C`; the ports and the YARP network are closed cleanly on
exit.

### Tuning

These are set in code (top of the file / `init()`):

- `max_depth` (default `8.0`) — the metric range mapped to `0–255`. Match it to
  the depth output produced by `depthize_dataset.py` so training and runtime
  agree.
- `token_number` (default `3000`) — number of ViT tokens per frame. The model's
  range is `[1200, 3600]`; more tokens = more accurate but slower. Lower it to
  speed inference up.

---

## `depthize_dataset.py`

Offline tool that takes an existing LeRobot dataset, runs MoGe on its RGB
frames, and writes a **new** dataset where the depth feature holds the
MoGe-estimated depth. The result is finalized and pushed to the Hugging Face
Hub as a private repo.

By default it reads `R1-HSP/approach-pick-plush` and writes
`R1-HSP/approach-pick-plush-depth-estimated`.

### Requirements

- A Hugging Face token with read access to the source repo and write access to
  your target repo, exported as an environment variable:

  ```bash
  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```

  The script calls `huggingface_hub.login(token=os.environ["HF_TOKEN"])`, so it
  fails if `HF_TOKEN` is unset.
- LeRobot installed (`pip install -e '.[dataset]'`).
- A CUDA GPU.

### Configure

Edit the constants near the top of the file before running:

| Constant | Default | Meaning |
|----------|---------|---------|
| `SOURCE_REPO_ID` | `R1-HSP/approach-pick-plush` | Dataset to read |
| `TARGET_REPO_ID` | `<source>-depth-estimated` | Dataset to create/push |
| `DEPTH_KEY` | `observation.images.egocentric_depth` | Feature replaced with the estimated depth |
| `BATCH_SIZE` | `8` | Frames per GPU forward pass — raise until VRAM is nearly full |
| `NUM_WORKERS` | `8` | CPU workers decoding video ahead of the GPU — set to ~physical core count (video decode is the bottleneck) |
| `NUM_TOKENS` | `1800` | ViT tokens per frame — dominant speed knob; lower = faster, less detail |
| `MAX_DEPTH_M` | `8.0` | Metric range (m) mapped to `0–255`; pixels beyond it saturate at `255` |
| `vis` | `False` | If `True`, shows RGB + depth side by side per frame (debug only) |

> Keep `MAX_DEPTH_M` here equal to `max_depth` in `depth_runtime.py` so the
> encoded depth has the same meaning offline and online.

### Run

```bash
export HF_TOKEN=hf_...
python moge/scripts/depthize_dataset.py
```

It processes the dataset episode by episode (progress bars via `tqdm`), saving
each episode as it goes, then finalizes and pushes the new dataset to
`TARGET_REPO_ID` (private). When it finishes you'll see
`Done. Dataset pushed to <TARGET_REPO_ID>`.

### Notes

- The depth is stored as a 3-channel image (the grayscale depth replicated
  across R/G/B) under `DEPTH_KEY`, encoded with the same `[0, MAX_DEPTH_M] →
  0–255` mapping as the runtime script.
- The script sets `torch.multiprocessing` sharing strategy to `file_system` to
  avoid `/dev/shm` overflow ("unable to allocate shared memory") that is common
  inside containers with a small shared-memory mount. If you run it in Docker,
  start the container with a large/host IPC (`--ipc=host`, already set in
  `run_lerobot.sh`).

---

## Docker

The [`docker/`](../../docker) folder provides a self-contained image (CUDA 12.8
+ Ubuntu 24.04) with ROS 2 Jazzy, YARP (with Python bindings), LeRobot and MoGe
already installed, so both scripts above run out of the box.

### Build

The build needs your Git identity and a GitHub token (the image clones the
private `hsp-iit/lerobot` repo). Use the helper script from inside the `docker/`
folder:

```bash
cd docker
./build_lerobot_moge_docker.sh "<git-username>" "<git-email>" "<github-token>"
```

This produces the image `lerobot_moge:latest`. (Internally it runs
`docker build` passing the three values as `--build-arg`s, and applies
`fix_swig_yarp.sh`, which swaps the buggy Ubuntu 24.04 SWIG 4.2.0 for SWIG 4.3.0
so the YARP Python bindings build correctly.)

### Run

```bash
cd docker
./run_lerobot.sh
```

This starts an interactive container (`bash`) with:

- `--gpus all` — GPU access (required by both scripts).
- `--network=host` + `--ipc=host` — host networking (so YARP ports are visible
  outside the container) and host shared memory (needed by the LeRobot
  DataLoader workers).
- X11 forwarding (`DISPLAY`, `/tmp/.X11-unix`, `/dev/dri/card0`) — lets GUI
  tools (`yarpview`, matplotlib `vis`) display on the host.
- `-v /home/ergocub/rosbags:/home/user1/rosbags` — host folder mounted into the
  container. Edit this path to share your own data.
- `ROS_DOMAIN_ID=65` — adjust if it clashes with other ROS 2 nodes on the
  network.

The container is removed on exit (`--rm`). Inside it, MoGe lives at
`/home/user1/MoGe`, so you can run:

```bash
# real-time
python3 ~/MoGe/moge/scripts/depth_runtime.py

# dataset batch (needs HF_TOKEN)
export HF_TOKEN=hf_...
python3 ~/MoGe/moge/scripts/depthize_dataset.py
```

> `run_lerobot.sh` calls `sudo xhost +` to allow the container to reach your X
> server. This disables X access control on the host — fine on a personal
> workstation, but be aware of it on shared machines.
