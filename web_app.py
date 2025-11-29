"""
Simple Gradio web application to run SAM 3D Objects inference.

Usage:
    python web_app.py

The app expects that checkpoints are available locally under
`checkpoints/<tag>/pipeline.yaml` (defaults to `hf`).
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict

import gradio as gr
import numpy as np
from PIL import Image

# Import inference utilities from the notebook package
sys.path.append("notebook")
from inference import Inference  # type: ignore  # noqa: E402

DEFAULT_TAG = "hf"
MODEL_CACHE: Dict[str, Inference] = {}


def get_inference(tag: str) -> Inference:
    """Load (and cache) an inference pipeline for the provided tag."""
    if tag not in MODEL_CACHE:
        config_path = Path("checkpoints") / tag / "pipeline.yaml"
        if not config_path.exists():
            raise gr.Error(
                f"Config not found at {config_path}. Make sure checkpoints are downloaded."
            )
        MODEL_CACHE[tag] = Inference(str(config_path), compile=False)
    return MODEL_CACHE[tag]


def _image_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _mask_to_numpy(mask: Image.Image) -> np.ndarray:
    # Convert to grayscale and threshold
    mask_array = np.array(mask.convert("L"))
    return mask_array > 0


def run_inference(image: Image.Image, mask: Image.Image, seed: int, tag: str):
    if image is None:
        raise gr.Error("Please upload an image to reconstruct.")
    if mask is None:
        raise gr.Error("Please upload a binary mask for the target object.")

    inference = get_inference(tag)

    image_np = _image_to_numpy(image)
    mask_np = _mask_to_numpy(mask)

    result = inference(image_np, mask_np, seed=seed)

    ply_path = Path(tempfile.mkdtemp()) / "reconstruction.ply"
    result["gs"].save_ply(str(ply_path))

    message = (
        "Reconstruction complete. Download the PLY file and open it in a 3D viewer."
    )
    return str(ply_path), message


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="SAM 3D Objects Web Demo") as demo:
        gr.Markdown(
            """
            # SAM 3D Objects Web Demo
            Upload an input image and its binary mask to run the SAM 3D Objects model.\
            The mask should highlight the object you want to reconstruct.
            """
        )

        with gr.Row():
            image_input = gr.Image(label="Input image", type="pil")
            mask_input = gr.Image(label="Mask (white = foreground)", type="pil")

        with gr.Row():
            seed = gr.Slider(0, 100000, value=42, step=1, label="Random seed")
            tag = gr.Textbox(value=DEFAULT_TAG, label="Checkpoint tag")

        run_button = gr.Button("Run reconstruction", variant="primary")

        with gr.Row():
            ply_output = gr.File(label="Gaussian splat (PLY)")
            status = gr.Markdown()

        run_button.click(
            fn=run_inference,
            inputs=[image_input, mask_input, seed, tag],
            outputs=[ply_output, status],
        )

    return demo


def main():
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
