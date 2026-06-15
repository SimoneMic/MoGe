import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
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

# Number of base ViT tokens per frame. This is the dominant speed knob for the
# ViT-L backbone (attention cost grows with token count). The model's range is
# [1200, 3600]; the default resolution_level=9 uses the max (3600), which is
# above the docstring's suggested 1200-2500. Lowering this is the cheapest real
# speedup, trading fine depth detail for throughput. Tune for your quality bar.
NUM_TOKENS = 1800

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

    episode_frames = [source[i] for i in range(from_idx, to_idx)]
    n_frames = len(episode_frames)

    # Run MoGe inference on every frame and collect raw depths
    depths = []
    frame_bar = tqdm(episode_frames, desc=f"  Ep {ep_idx:>3} frames", unit="fr", leave=False)
    for frame in frame_bar:
        rgb = frame[rgb_key].to(device)          # (C, H, W) float32 in [0, 1]
        with torch.inference_mode():
            output = model.infer(rgb, num_tokens=NUM_TOKENS)
        depths.append(output["depth"].cpu().numpy())   # (H, W) metric scale

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
