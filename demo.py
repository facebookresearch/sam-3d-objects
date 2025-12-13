# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image (RGBA only, mask is embedded in the alpha channel)
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# Optional USD export path (set to a filename to enable)
usd_path = "reconstruction.usd"  # e.g., "reconstruction.usd"
# Adjust the scale factor to match your scene units (default 100)
usd_scale_factor = 100.0

# run model
output = inference(image, mask, seed=42, export_usd_path=usd_path, usd_scale_factor=usd_scale_factor)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")
if output.get("usd_path"):
    print(f"USD mesh with texture saved to {output['usd_path']}")
elif usd_path is not None:
    print("USD export requested but failed; check logs for details.")
