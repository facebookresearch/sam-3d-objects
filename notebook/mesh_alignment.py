# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
SAM 3D Body (3DB) Mesh Alignment Utilities
Handles alignment of 3DB meshes to SAM 3D Object, same as MoGe point cloud scale.
"""

import os
import math
import json
import numpy as np
import torch
import trimesh
from PIL import Image
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer, TexturesVertex
from moge.model.v1 import MoGeModel


def load_3db_mesh(mesh_path, device='cuda'):
    """Load 3DB mesh and convert from OpenGL to PyTorch3D coordinates."""
    mesh = trimesh.load(mesh_path)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Convert from OpenGL to PyTorch3D coordinates
    vertices[:, 0] *= -1  # Flip X
    vertices[:, 2] *= -1  # Flip Z

    vertices = torch.from_numpy(vertices).float().to(device)
    faces = torch.from_numpy(faces).long().to(device)
    return vertices, faces


def get_moge_pointcloud(image_tensor, device='cuda'):
    """Generate MoGe point cloud from image tensor."""
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    moge_model.eval()
    with torch.no_grad():
        moge_output = moge_model.infer(image_tensor)
    return moge_output


def denormalize_intrinsics(norm_K, height, width):
    """Convert normalized intrinsics to absolute pixel coordinates."""
    cx_norm, cy_norm = norm_K[0, 2], norm_K[1, 2]
    fx_norm, fy_norm = norm_K[0, 0], norm_K[1, 1]

    fx_abs = fx_norm * width
    fy_abs = fy_norm * height
    cx_abs = cx_norm * width
    cy_abs = cy_norm * height
    fx_abs = fy_abs 

    return np.array([
        [fx_abs, 0.0, cx_abs],
        [0.0, fy_abs, cy_abs],
        [0.0, 0.0, 1.0]
    ])


def crop_mesh_with_mask(vertices, faces, focal_length, mask, device='cuda'):
    """Crop mesh vertices to only those visible in the mask."""
    textures = TexturesVertex(verts_features=torch.ones_like(vertices)[None])
    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)

    H, W = mask.shape[-2:]
    fx = fy = focal_length
    cx, cy = W / 2.0, H / 2.0

    camera = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        image_size=((H, W),),
        in_ndc=False, device=device
    )

    raster_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1,
        cull_backfaces=False, bin_size=0,
    )

    rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)
    fragments = rasterizer(mesh)

    face_indices = fragments.pix_to_face[0, ..., 0]  # (H, W)
    visible_mask = (mask > 0) & (face_indices >= 0)
    visible_face_ids = face_indices[visible_mask]

    visible_faces = faces[visible_face_ids]
    visible_vert_ids = torch.unique(visible_faces)
    verts_cropped = vertices[visible_vert_ids]

    return verts_cropped, visible_mask


def extract_target_points(pointmap, visible_mask):
    """Extract target points from MoGe pointmap using visible mask."""
    target_points = pointmap[visible_mask.bool()]

    # Convert from MoGe coordinates to PyTorch3D coordinates
    target_points[:, 0] *= -1
    target_points[:, 1] *= -1

    # Remove flying points using adaptive quantile filtering
    z_range = torch.max(target_points[:, 2]) - torch.min(target_points[:, 2])
    if z_range > 6.0:
        thresh = 0.90
    elif z_range > 2.0:
        thresh = 0.93
    else:
        thresh = 0.95

    depth_quantile = torch.quantile(target_points[:, 2], thresh)
    target_points = target_points[target_points[:, 2] <= depth_quantile]

    # Remove infinite values
    finite_mask = torch.isfinite(target_points).all(dim=1)
    target_points = target_points[finite_mask]

    return target_points


def align_mesh_to_pointcloud(vertices, target_points):
    """Align mesh vertices to target point cloud using scale and translation."""
    if target_points.shape[0] == 0:
        print("[WARNING] No target points for alignment!")
        return vertices, torch.tensor(1.0), torch.zeros(3)

    # Scale alignment based on height
    height_src = torch.max(vertices[:, 1]) - torch.min(vertices[:, 1])
    height_tgt = torch.max(target_points[:, 1]) - torch.min(target_points[:, 1])
    scale_factor = height_tgt / height_src

    vertices_scaled = vertices * scale_factor

    # Translation alignment based on centers
    center_src = torch.mean(vertices_scaled, dim=0)
    center_tgt = torch.mean(target_points, dim=0)
    translation = center_tgt - center_src

    vertices_aligned = vertices_scaled + translation
    return vertices_aligned, scale_factor, translation


def load_mask_for_alignment(mask_path):
    """Load mask image as numpy array."""
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask) / 255.0
    return mask_array


def load_focal_length_from_json(json_path):
    """Load focal length from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        focal_length = data.get('focal_length')
        if focal_length is None:
            raise ValueError("'focal_length' key not found in JSON file")
        print(f"[INFO] Loaded focal length from {json_path}: {focal_length}")
        return focal_length
    except Exception as e:
        print(f"[ERROR] Failed to load focal length from {json_path}: {e}")
        raise


def process_3db_alignment(mesh_path, mask_path, image_path, device='cuda', focal_length_json_path=None):
    """Complete pipeline for aligning 3DB mesh to MoGe scale."""
    print(f"[INFO] Processing alignment...")

    # Load input data
    vertices, faces = load_3db_mesh(mesh_path, device)

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.to(device)

    # Load mask and resize to match image
    H, W = image_tensor.shape[1:]
    mask = load_mask_for_alignment(mask_path)
    if mask.shape != (H, W):
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        mask = mask.resize((W, H), Image.NEAREST)
        mask = np.array(mask) / 255.0
    mask = torch.from_numpy(mask).float().to(device)

    # Generate MoGe point cloud
    print("[INFO] Generating MoGe point cloud...")
    moge_output = get_moge_pointcloud(image_tensor, device)

    # Load focal length from JSON if provided, otherwise compute from MoGe intrinsics
    if focal_length_json_path is not None:
        focal_length = load_focal_length_from_json(focal_length_json_path)
    else:
        # Compute camera parameters from MoGe intrinsics (fallback)
        intrinsics = denormalize_intrinsics(moge_output['intrinsics'].cpu().numpy(), H, W)
        focal_length = intrinsics[1, 1]  # Use fy
        print(f"[INFO] Using computed focal length from MoGe: {focal_length}")

    # Crop mesh using mask
    print("[INFO] Cropping mesh with mask...")
    verts_cropped, visible_mask = crop_mesh_with_mask(vertices, faces, focal_length, mask, device)

    # Extract target points from MoGe
    print("[INFO] Extracting target points...")
    target_points = extract_target_points(moge_output['points'], visible_mask)

    if target_points.shape[0] == 0:
        print("[ERROR] No valid target points found!")
        return None

    # Perform alignment
    print("[INFO] Aligning mesh to point cloud...")
    aligned_vertices, scale_factor, translation = align_mesh_to_pointcloud(verts_cropped, target_points)

    # Apply alignment to full mesh
    full_aligned_vertices = (vertices * scale_factor) + translation

    # Convert back to OpenGL coordinates for final output
    final_vertices_opengl = full_aligned_vertices.cpu().numpy()
    final_vertices_opengl[:, 0] *= -1
    final_vertices_opengl[:, 2] *= -1

    results = {
        'aligned_vertices_opengl': final_vertices_opengl,
        'faces': faces.cpu().numpy(),
        'scale_factor': scale_factor.item(),
        'translation': translation.cpu().numpy(),
        'focal_length': focal_length,
        'target_points_count': target_points.shape[0],
        'cropped_vertices_count': verts_cropped.shape[0]
    }

    print(f"[INFO] Alignment completed - Scale: {scale_factor.item():.4f}, Target points: {target_points.shape[0]}")
    return results


def process_and_save_alignment(mesh_path, mask_path, image_path, output_dir, device='cuda', focal_length_json_path=None):
    """
    Complete pipeline for processing 3DB alignment and saving the result.

    Args:
        mesh_path: Path to input 3DB mesh (.ply)
        mask_path: Path to mask image (.png)
        image_path: Path to input image (.jpg)
        output_dir: Directory to save aligned mesh
        device: Device to use ('cuda' or 'cpu')
        focal_length_json_path: Optional path to focal length JSON file

    Returns:
        tuple: (success: bool, output_mesh_path: str or None, result_info: dict or None)
    """
    try:
        print("[INFO] Starting 3DB mesh alignment pipeline...")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Process alignment
        result = process_3db_alignment(
            mesh_path=mesh_path,
            mask_path=mask_path,
            image_path=image_path,
            device=device,
            focal_length_json_path=focal_length_json_path
        )

        if result is not None:
            # Save aligned mesh
            output_mesh_path = os.path.join(output_dir, 'human_aligned.ply')
            aligned_mesh = trimesh.Trimesh(
                vertices=result['aligned_vertices_opengl'],
                faces=result['faces']
            )
            aligned_mesh.export(output_mesh_path)

            print(f" SUCCESS! Saved aligned mesh to: {output_mesh_path}")
            return True, output_mesh_path, result
        else:
            print(" ERROR: Failed to process mesh alignment")
            return False, None, None

    except Exception as e:
        print(f" ERROR: Exception during processing: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

    finally:
        print(" Processing complete!")


def visualize_meshes_interactive(aligned_mesh_path, dfy_mesh_path, output_dir=None, share=True, height=600):
    """
    Interactive Gradio-based 3D visualization of aligned human and object meshes.

    Args:
        aligned_mesh_path: Path to aligned mesh PLY file
        dfy_mesh_path: Path to 3Dfy Object file
        output_dir: Directory to save combined GLB file (defaults to same dir as aligned_mesh_path)
        share: Whether to create a public shareable link (default: True)
        height: Height of the 3D viewer in pixels (default: 600)

    Returns:
        tuple: (demo, combined_glb_path) - Gradio demo object and path to combined GLB file
    """
    import gradio as gr
    import numpy as np

    print("Loading meshes for interactive visualization...")

    try:
        # Load aligned mesh (PLY)
        aligned_mesh = trimesh.load(aligned_mesh_path)
        print(f"Loaded aligned mesh: {len(aligned_mesh.vertices)} vertices")

        # Load 3Dfy mesh (PLY)
        dfy_scene = trimesh.load(dfy_mesh_path)

        if hasattr(dfy_scene, 'dump'):
            dfy_meshes = [geom for geom in dfy_scene.geometry.values() if hasattr(geom, 'vertices')]
            if len(dfy_meshes) == 1:
                dfy_mesh = dfy_meshes[0]
            elif len(dfy_meshes) > 1:
                dfy_mesh = trimesh.util.concatenate(dfy_meshes)
            else:
                raise ValueError("No valid meshes in PLY file")
        else:
            dfy_mesh = dfy_scene

        print(f"Loaded 3Dfy mesh: {len(dfy_mesh.vertices)} vertices")

        # Create combined scene
        scene = trimesh.Scene()

        # Add both meshes with different colors
        aligned_copy = aligned_mesh.copy()
        aligned_copy.visual.vertex_colors = [255, 0, 0, 200]  # Red for aligned human
        scene.add_geometry(aligned_copy, node_name="sam3d_aligned_human")

        dfy_copy = dfy_mesh.copy()
        dfy_copy.visual.vertex_colors = [0, 0, 255, 200]  # Blue for 3Dfy object
        scene.add_geometry(dfy_copy, node_name="dfy_object")

        # Determine output path
        if output_dir is None:
            output_dir = os.path.dirname(aligned_mesh_path)
        os.makedirs(output_dir, exist_ok=True)

        # Save combined PLY by concatenating both meshes
        combined_ply_path = os.path.join(output_dir, 'combined_scene.ply')
        
        # Ccombine the geometries for PLY output
        if isinstance(dfy_mesh, trimesh.points.PointCloud):
            # Convert point cloud to vertices-only mesh for combination
            dfy_vertices = dfy_mesh.vertices
            human_vertices = aligned_mesh.vertices
            
            # Combine vertices from both
            all_vertices = np.vstack([human_vertices, dfy_vertices])
            
            # Create colors: red for human, blue for object
            human_colors = np.array([[255, 0, 0, 200]] * len(human_vertices))
            object_colors = np.array([[0, 0, 255, 200]] * len(dfy_vertices))
            all_colors = np.vstack([human_colors, object_colors])
            
            # Create combined point cloud
            combined_cloud = trimesh.points.PointCloud(vertices=all_vertices, colors=all_colors)
            combined_cloud.export(combined_ply_path)
        else:
            # Both are meshes, use scene export
            scene.export(combined_ply_path)
        
        print(f"Exported combined scene to: {combined_ply_path}")

        # Also save GLB for Gradio viewer (NOTE: GLB may not show point cloud object properly)
        combined_glb_path = os.path.join(output_dir, 'combined_scene.glb')
        scene.export(combined_glb_path)
        print(f"Exported GLB for Gradio viewer to: {combined_glb_path}")
        print("NOTE: Use PLY for complete data, GLB is for Gradio visualization only")

        # Create interactive Gradio viewer
        with gr.Blocks() as demo:
            gr.Markdown("# 3D Mesh Alignment Visualization")
            gr.Markdown("**Red**: SAM 3D Body Aligned Human | **Blue**: SAM 3D Object")
            gr.Model3D(
                value=combined_glb_path,
                label="Combined 3D Scene (Interactive)",
                height=height
            )

        # Launch the viewer
        print("Launching interactive 3D viewer...")
        demo.launch(share=share)

        return demo, combined_glb_path

    except Exception as e:
        print(f"ERROR in visualization: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def visualize_meshes_comparison(aligned_mesh_path, dfy_mesh_path, use_interactive=False):
    """
    Simple visualization of both meshes in a single 3D plot.

    DEPRECATED: Use visualize_meshes_interactive() for better interactive visualization.

    Args:
        aligned_mesh_path: Path to aligned mesh PLY file
        dfy_mesh_path: Path to 3Dfy GLB file
        use_interactive: Whether to attempt trimesh scene viewer (default: False)

    Returns:
        tuple: (aligned_mesh, dfy_mesh) trimesh objects or (None, None) if failed
    """
    import matplotlib.pyplot as plt

    print("Loading meshes for visualization...")

    try:
        # Load aligned mesh (PLY)
        aligned_mesh = trimesh.load(aligned_mesh_path)
        print(f"Loaded aligned mesh: {len(aligned_mesh.vertices)} vertices")

        # Load 3Dfy mesh (GLB - handle scene structure)
        dfy_scene = trimesh.load(dfy_mesh_path)

        if hasattr(dfy_scene, 'dump'):  # It's a scene
            dfy_meshes = [geom for geom in dfy_scene.geometry.values() if hasattr(geom, 'vertices')]
            if len(dfy_meshes) == 1:
                dfy_mesh = dfy_meshes[0]
            elif len(dfy_meshes) > 1:
                dfy_mesh = trimesh.util.concatenate(dfy_meshes)
            else:
                raise ValueError("No valid meshes in GLB file")
        else:
            dfy_mesh = dfy_scene

        print(f"Loaded 3Dfy mesh: {len(dfy_mesh.vertices)} vertices")

        # Create single 3D plot with both meshes
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot both meshes in the same space
        ax.scatter(dfy_mesh.vertices[:, 0],
                   dfy_mesh.vertices[:, 1],
                   dfy_mesh.vertices[:, 2],
                   c='blue', s=0.1, alpha=0.6, label='3Dfy Original')

        ax.scatter(aligned_mesh.vertices[:, 0],
                   aligned_mesh.vertices[:, 1],
                   aligned_mesh.vertices[:, 2],
                   c='red', s=0.1, alpha=0.6, label='SAM 3D Body Aligned')

        ax.set_title('Mesh Comparison: 3Dfy vs SAM 3D Body Aligned', fontsize=16, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.tight_layout()
        plt.show()

        # Optional trimesh scene viewer
        if use_interactive:
            try:
                print("Creating trimesh scene...")
                scene = trimesh.Scene()

                # Add both meshes with different colors
                aligned_copy = aligned_mesh.copy()
                aligned_copy.visual.vertex_colors = [255, 0, 0, 200]  # Red
                scene.add_geometry(aligned_copy, node_name="sam3d_aligned")

                dfy_copy = dfy_mesh.copy()
                dfy_copy.visual.vertex_colors = [0, 0, 255, 200]  # Blue
                scene.add_geometry(dfy_copy, node_name="dfy_original")

                print("Opening interactive trimesh viewer...")
                scene.show()

            except Exception as e:
                print(f"Trimesh viewer not available: {e}")

        print("Visualization complete")
        return aligned_mesh, dfy_mesh

    except Exception as e:
        print(f"ERROR in visualization: {e}")
        import traceback
        traceback.print_exc()
        return None, None