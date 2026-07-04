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
image = load_image("/work/goet1019/ai_art/proposal.jpg")
#mask1 = load_single_mask("notebook/images/bike/sam3_image", index=0)
mask2 = load_single_mask("/work/goet1019/ai_art", index=0)

# run model
#output1 = inference(image, mask1, seed=42)
output2 = inference(image, mask2, seed=42)
print("=== Output Keys ===")
print("=== Output Details ===")
for key, value in output2.items():
    if hasattr(value, 'shape'):
        print(f"{key:25} | Type: {type(value).__name__:15} | Shape: {value.shape}")
    
    elif isinstance(value, (int, float, list)):
        print(f"{key:25} | Type: {type(value).__name__:15} | Value: {value}")
    
    else:
        print(f"{key:25} | Type: {type(value).__name__:15} | Info: {value}")
print("======================")


# export gaussian splat
output2["gs"].save_ply(f"ai_art.ply")


