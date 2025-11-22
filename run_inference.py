"""
SAM 3D Objects Inference Script
Supports both single-view and multi-view 3D reconstruction

Usage:
    # Multi-view inference (mask_prompt=None, images and masks in same directory, use all images)
    python run_inference.py --input_path ./data/images_and_masks
    
    # Single-view inference (specify a single image name)
    python run_inference.py --input_path ./data/images_and_masks --image_names image1
    
    # Multi-view inference (mask_prompt!=None, images in images/, masks in specified folder)
    python run_inference.py --input_path ./data --mask_prompt stuffed_toy
    
    # Specify multiple image names (can be any filename without extension)
    python run_inference.py --input_path ./data --mask_prompt stuffed_toy --image_names image1,view_a,2
"""
import sys
import argparse
from pathlib import Path
from typing import List, Optional
from loguru import logger

# 导入推理代码
sys.path.append("notebook")
from inference import Inference
from load_images_and_masks import load_images_and_masks_from_path


def parse_image_names(image_names_str: Optional[str]) -> Optional[List[str]]:
    """
    Parse image names string
    
    Args:
        image_names_str: Image names string, e.g., "image1,view_a" or "1,2" or "image1"
                         Can be any filename (without extension) or numbers
    
    Returns:
        image_names: List of image names (without extension), None means use all available images
    """
    if image_names_str is None or image_names_str == "":
        return None
    
    names = [x.strip() for x in image_names_str.split(",") if x.strip()]
    return names if names else None


def get_output_dir(
    input_path: Path, 
    mask_prompt: Optional[str] = None, 
    image_names: Optional[List[str]] = None,
    is_single_view: bool = False
) -> Path:
    """
    Create output directory based on input path and parameters
    
    Args:
        input_path: Input path
        mask_prompt: Mask folder name (if using separated directory structure)
        image_names: List of image names
        is_single_view: Whether it's single-view inference
    
    Returns:
        output_dir: Path to visualization/{mask_prompt_or_dirname}_{image_names}/ directory
    """
    visualization_dir = Path("visualization")
    visualization_dir.mkdir(exist_ok=True)
    
    if mask_prompt:
        dir_name = mask_prompt
    else:
        dir_name = input_path.name if input_path.is_dir() else input_path.parent.name
    
    if is_single_view:
        if image_names and len(image_names) == 1:
            safe_name = image_names[0].replace("/", "_").replace("\\", "_")
            dir_name = f"{dir_name}_{safe_name}"
        else:
            dir_name = f"{dir_name}_single"
    elif image_names:
        if len(image_names) == 1:
            safe_name = image_names[0].replace("/", "_").replace("\\", "_")
            dir_name = f"{dir_name}_{safe_name}"
        else:
            safe_names = [name.replace("/", "_").replace("\\", "_") for name in image_names]
            dir_name = f"{dir_name}_{'_'.join(safe_names[:3])}"
            if len(safe_names) > 3:
                dir_name += f"_and_{len(safe_names)-3}_more"
    else:
        dir_name = f"{dir_name}_multiview"
    
    output_dir = visualization_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    return output_dir


def run_inference(
    input_path: Path,
    mask_prompt: Optional[str] = None,
    image_names: Optional[List[str]] = None,
    seed: int = 42,
    stage1_steps: int = 50,
    stage2_steps: int = 25,
    decode_formats: List[str] = None,
    model_tag: str = "hf",
):
    """
    Run inference
    
    Args:
        input_path: Input path
        mask_prompt: Mask folder name, if None then images and masks are in the same directory
        image_names: List of image names (without extension), e.g., ["image1", "view_a"] or ["1", "2"],
                     None means use all available images
        seed: Random seed
        stage1_steps: Stage 1 inference steps
        stage2_steps: Stage 2 inference steps
        decode_formats: List of decode formats
        model_tag: Model tag
    """
    config_path = f"checkpoints/{model_tag}/pipeline.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    
    logger.info(f"Loading model: {config_path}")
    inference = Inference(config_path, compile=False)
    
    if hasattr(inference._pipeline, 'rendering_engine'):
        if inference._pipeline.rendering_engine != "pytorch3d":
            logger.warning(f"Rendering engine is set to {inference._pipeline.rendering_engine}, changing to pytorch3d")
            inference._pipeline.rendering_engine = "pytorch3d"
    
    logger.info(f"Loading data: {input_path}")
    if mask_prompt:
        logger.info(f"Mask prompt: {mask_prompt} (images in images/, masks in {mask_prompt}/)")
    else:
        logger.info("Mask prompt: None (images and masks in same directory)")
    
    view_images, view_masks = load_images_and_masks_from_path(
        input_path=input_path,
        mask_prompt=mask_prompt,
        image_names=image_names,
    )
    
    num_views = len(view_images)
    logger.info(f"Successfully loaded {num_views} views")
    
    is_single_view = num_views == 1
    
    if is_single_view:
        logger.info("Single-view inference mode")
        image = view_images[0]
        mask = view_masks[0] if view_masks else None
        result = inference(image, mask, seed=seed)
    else:
        logger.info("Multi-view inference mode")
        decode_formats = decode_formats or ["gaussian", "mesh"]
        
        result = inference._pipeline.run_multi_view(
            view_images=view_images,
            view_masks=view_masks,
            seed=seed,
            mode="multidiffusion",
            stage1_inference_steps=stage1_steps,
            stage2_inference_steps=stage2_steps,
            decode_formats=decode_formats,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
        )
    
    output_dir = get_output_dir(input_path, mask_prompt, image_names, is_single_view)
    saved_files = []
    
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Generated coordinates: {result['coords'].shape[0] if 'coords' in result else 'N/A'}")
    print(f"{'='*60}")
    
    if 'glb' in result and result['glb'] is not None:
        output_path = output_dir / "result.glb"
        result['glb'].export(str(output_path))
        saved_files.append("result.glb")
        print(f"✓ GLB file saved to: {output_path}")
    
    if 'gs' in result:
        output_path = output_dir / "result.ply"
        result['gs'].save_ply(str(output_path))
        saved_files.append("result.ply")
        print(f"✓ Gaussian Splatting (PLY) saved to: {output_path}")
    elif 'gaussian' in result:
        if isinstance(result['gaussian'], list) and len(result['gaussian']) > 0:
            output_path = output_dir / "result.ply"
            result['gaussian'][0].save_ply(str(output_path))
            saved_files.append("result.ply")
            print(f"✓ Gaussian Splatting (PLY) saved to: {output_path}")
    
    if 'mesh' in result:
        print(f"✓ Mesh information generated (included in GLB)")
    
    print(f"\n{'='*60}")
    print(f"All output files saved to: {output_dir}")
    print(f"Saved files: {', '.join(saved_files)}")
    print(f"{'='*60}")
    
    print(f"\nFile descriptions:")
    print(f"- PLY file: Gaussian Splatting format with position and color information")
    print(f"  * Recommended to use specialized Gaussian Splatting viewers")
    print(f"- GLB file: Complete 3D mesh model, can be viewed in Blender, Three.js, etc.")


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D Objects Inference Script - Supports single-view and multi-view 3D reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-view inference (mask_prompt=None, images and masks in same directory, use all images)
  python run_inference.py --input_path ./data/images_and_masks
  
  # Single-view inference (specify a single image name)
  python run_inference.py --input_path ./data/images_and_masks --image_names image1
  
  # Multi-view inference (mask_prompt!=None, images in images/, masks in specified folder)
  python run_inference.py --input_path ./data --mask_prompt stuffed_toy
  
  # Specify multiple image names (can be any filename without extension)
  python run_inference.py --input_path ./data --mask_prompt stuffed_toy --image_names image1,view_a,2
        """
    )
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input path. If mask_prompt=None, images and masks are in this directory; "
             "if mask_prompt!=None, images are in input_path/images/, masks in input_path/{mask_prompt}/"
    )
    parser.add_argument(
        "--mask_prompt",
        type=str,
        default=None,
        help="Mask folder name. If None, images and masks are in the same directory "
             "(naming format: xxxx.png and xxxx_mask.png); "
             "if not None, images are in input_path/images/, masks in input_path/{mask_prompt}/"
    )
    parser.add_argument(
        "--image_names",
        type=str,
        default=None,
        help="Image names (without extension), e.g., 'image1,view_a' or '1,2' or 'image1'. "
             "Can specify multiple, comma-separated. If not specified, use all available images"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--stage1_steps",
        type=int,
        default=50,
        help="Stage 1 inference steps (default: 50)"
    )
    parser.add_argument(
        "--stage2_steps",
        type=int,
        default=25,
        help="Stage 2 inference steps (default: 25)"
    )
    
    parser.add_argument(
        "--decode_formats",
        type=str,
        default="gaussian,mesh",
        help="Decode formats, comma-separated, e.g., 'gaussian,mesh' or 'gaussian' (default: gaussian,mesh)"
    )
    
    parser.add_argument(
        "--model_tag",
        type=str,
        default="hf",
        help="Model tag (default: hf)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    image_names = parse_image_names(args.image_names)
    
    decode_formats = [fmt.strip() for fmt in args.decode_formats.split(",") if fmt.strip()]
    if not decode_formats:
        decode_formats = ["gaussian", "mesh"]
    
    try:
        run_inference(
            input_path=input_path,
            mask_prompt=args.mask_prompt,
            image_names=image_names,
            seed=args.seed,
            stage1_steps=args.stage1_steps,
            stage2_steps=args.stage2_steps,
            decode_formats=decode_formats,
            model_tag=args.model_tag,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
