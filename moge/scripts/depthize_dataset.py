import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# DataLoader workers pass decoded frames to the main process via /dev/shm by
# default, which is tiny in most containers and overflows ("unable to allocate
# shared memory"). The file_system strategy uses regular temp files instead.
mp.set_sharing_strategy("file_system")
from huggingface_hub import login
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES

# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel  # Let's try MoGe-2
import os
TOKEN=os.environ.get("HF_TOKEN")
login(token=TOKEN)

SOURCE_REPO_ID = "R1-HSP/approach-pick-plush"
TARGET_REPO_ID = SOURCE_REPO_ID + "-depth-estimated"
DEPTH_KEY = "observation.images.egocentric_depth"

vis = False

# Frames processed per GPU forward pass. Increase until VRAM is ~full
# (you have ~11 GB of headroom at the current single-frame usage).
BATCH_SIZE = 1
# Parallel CPU workers that decode video frames ahead of the GPU. This is the
# real bottleneck here: video decode is sequential CPU work that the GPU waits
# on. Set to ~number of physical cores; 0 disables (single-process decoding).
NUM_WORKERS = 0

device = torch.device("cuda")

model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(device)

# Load source dataset 
source = LeRobotDataset(SOURCE_REPO_ID)

print(f"Source dataset: {len(source)} frames, {source.num_episodes} episodes")

image_keys = [k for k in source.features if "image" in k.lower()]
rgb_key = next(k for k in image_keys if k != DEPTH_KEY)
print(f"RGB key: {rgb_key}  |  Depth key to replace: {DEPTH_KEY}")

# Create target dataset (same features, depth key will hold MoGe estimates)
user_features = {k: v for k, v in source.features.items() if k not in DEFAULT_FEATURES}
target = LeRobotDataset.create(
    repo_id=TARGET_REPO_ID,
    fps=source.fps,
    features=user_features,
)

# Iterate episode by episode
ep_bar = tqdm(range(source.num_episodes), desc="Episodes", unit="ep")
for ep_idx in ep_bar:
    from_idx = source.meta.episodes["dataset_from_index"][ep_idx]
    to_idx = source.meta.episodes["dataset_to_index"][ep_idx]

    indices = list(range(from_idx, to_idx + 1))
    n_frames = len(indices)

    # Decode frames in parallel worker processes and prefetch the next batches
    # while the GPU is busy. collate_fn keeps each batch as a list of frame
    # dicts (no stacking) so the save loop below works unchanged; order is
    # preserved because shuffle=False.
    loader = DataLoader(
        Subset(source, indices),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=lambda batch: batch,
        pin_memory=True,
        persistent_workers=False,
    )

    # Run MoGe inference in batches and collect raw depths + frames
    episode_frames = []
    depths = []
    batch_bar = tqdm(loader, desc=f"  Ep {ep_idx:>3} batches", unit="batch", leave=False)
    for batch_frames in batch_bar:
        rgb = torch.stack([f[rgb_key] for f in batch_frames]).to(device)  # (B, C, H, W) float32 in [0, 1]
        with torch.inference_mode():
            output = model.infer(rgb)
        depths.extend(output["depth"].cpu().numpy())   # B x (H, W) metric scale
        episode_frames.extend(batch_frames)

    # Normalise per episode to preserve relative depth across frames
    ep_min = min(d.min() for d in depths)
    ep_max = max(d.max() for d in depths)
    depth_range = ep_max - ep_min if ep_max > ep_min else 1.0

    for frame, depth in zip(episode_frames, depths):
        depth_u8 = ((depth - ep_min) / depth_range * 255).astype(np.uint8)  # (H, W)
        depth_rgb = np.stack([depth_u8, depth_u8, depth_u8], axis=-1)       # (H, W, 3)

        if vis:
            rgb_np = frame[rgb_key].permute(1, 2, 0).numpy()
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(rgb_np)
            axes[0].set_title(f"RGB — ep {ep_idx}")
            axes[0].axis("off")
            axes[1].imshow(depth_u8, cmap="turbo")
            axes[1].set_title(f"Estimated depth — ep {ep_idx}")
            axes[1].axis("off")
            plt.colorbar(axes[1].images[0], ax=axes[1], label="normalised depth")
            plt.tight_layout()
            plt.show()

        # add_frame expects images in (H, W, C); source tensors are (C, H, W)
        hw_frame = {
            k: (frame[k].permute(1, 2, 0).numpy()
                if k in source.meta.camera_keys else frame[k])
            for k in user_features
            if k != DEPTH_KEY
        }
        target.add_frame({
            **hw_frame,
            DEPTH_KEY: depth_rgb,
            "task": frame["task"],
        })

    target.save_episode()
    print(f"Episode {ep_idx + 1}/{source.num_episodes} saved.")

# Finalise and push
target.consolidate()
target.push_to_hub(private=True)
print(f"Done. Dataset pushed to {TARGET_REPO_ID}")
