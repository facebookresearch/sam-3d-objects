# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Multi-object 3D reconstruction demo script.
Processes multiple masks from an image and creates a combined 3D scene.
"""

import os
import sys
import argparse
import imageio
import numpy as np
import torch

from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf  # USD transforms
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

# Add notebook directory to path for imports
sys.path.append("notebook")
from inference import (
    Inference,
    ready_gaussian_for_video_rendering,
    load_image,
    load_masks,
    display_image,
    make_scene,
    render_video,
    interactive_visualizer,
)
from preprocess import preprocess


def apply_transforms_to_usd(
    usd_path: str,
    scale,
    usd_scale_factor: float = 1.0,
    index: int = 0,
):
    """
    Apply position and size transforms directly to an existing USD file.
    This wraps the mesh in an Xform with the same transforms that
    make_scene uses, so each USD file already has the correct world-space
    position and size baked in.

    Args:
        usd_path: Path to the USD file to modify
        scale: Scale factor (torch.Tensor or numpy array)
        usd_scale_factor: Scale factor used when exporting vertices
            (default 100.0)
        index: Index of the USD file
    """
    # ========================================================================
    # Pipeline Step 1: Load SAM-3D-Objects USD file
    # ========================================================================
    if not os.path.exists(usd_path):
        print(
            f"    Warning: USD file not found at {usd_path}, "
            "skipping transform application"
        )
        return

    # Open existing USD file
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(
            f"    Warning: Could not open USD file at {usd_path}, "
            "skipping transform application"
        )
        return

    # Find the mesh prim (usually at "/Mesh")
    mesh_prim = stage.GetPrimAtPath("/Mesh")
    print(f"  Found mesh at: {mesh_prim.GetPath()}")
    
    # ========================================================================
    # Pipeline Step 2: Create Xform and set it as default prim
    # ========================================================================
    # Read mesh data
    mesh_geom = UsdGeom.Mesh(mesh_prim)
    points = mesh_geom.GetPointsAttr().Get()
    face_vertex_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()
    
    # Remove old mesh and create new structure:
    # /Object (Xform) -> /Object/Mesh (Mesh)
    mesh_name = 'Mesh'
    stage.RemovePrim(f"/{mesh_name}")
    xform_prim = UsdGeom.Xform.Define(stage, f"/Object_{index}")
    new_mesh_prim = UsdGeom.Mesh.Define(
        stage, f"/Object_{index}/{mesh_name}_{index}"
    )
    
    # Copy mesh data
    new_mesh_prim.CreatePointsAttr(points)
    new_mesh_prim.CreateFaceVertexCountsAttr(face_vertex_counts)
    new_mesh_prim.CreateFaceVertexIndicesAttr(face_vertex_indices)

    # Get material and copy it into the new Xform.
    material_name = "Material"
    src_material = UsdShade.Material.Get(stage, Sdf.Path(f"/{material_name}"))
    if src_material:
        src_prim = src_material.GetPrim()
        src_path = src_prim.GetPath()
        # Path for the duplicated material
        dst_path = Sdf.Path(f"/Object_{index}/{material_name}_{index}")
        # Get the current edit layer
        layer = stage.GetEditTarget().GetLayer()
        # Copy the prim spec (and its children) from src to dst
        Sdf.CopySpec(layer, src_path, layer, dst_path)
        # Get the new material at the new path
        new_material = UsdShade.Material.Get(stage, dst_path)
        # Bind duplicated material to the new mesh prim
        if new_material:
            UsdShade.MaterialBindingAPI(new_mesh_prim).Bind(new_material)
        if not new_material:
            raise RuntimeError(f"Material at {dst_path} not found")

        shader_path = Sdf.Path(dst_path).AppendChild("Texture")
        shader = UsdShade.Shader.Get(stage, shader_path)
        if shader:
            shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        if not shader:
            raise RuntimeError(f"Shader at {shader_path} not found")
        # Delete the original material
        stage.RemovePrim(src_path)
    mesh_prim = new_mesh_prim

    # set default prim to xform_prim
    if xform_prim:
        stage.SetDefaultPrim(xform_prim.GetPrim())

    # # ========================================================================
    # # Pipeline Step 2: Insert scaled
    # # ========================================================================
    # Scale mesh vertices directly (like make_scene does to gaussians)
    if scale is not None:
        try:
            s = scale.detach().cpu().numpy().reshape(-1)
        except AttributeError:
            s = np.array(scale).reshape(-1)
        if s.size >= 1:
            s_val = float(s.mean())          
            mesh_geom = UsdGeom.Mesh(mesh_prim)
            points_attr = mesh_geom.GetPointsAttr()
            if points_attr:
                original_points = points_attr.Get()
                if original_points:
                    # Scale each point directly
                    scaled_points = [
                        Gf.Vec3f(p[0] * s_val, p[1] * s_val, p[2] * s_val)
                        for p in original_points
                    ]
                    points_attr.Set(scaled_points)

    # ========================================================================
    # Pipeline Step 3: Apply Rotation
    # ========================================================================
    # Apply rotation to convert from PLY (Y-up) to USD (Z-up) coordinate
    # system. Rotation: 90 degrees around X-axis to convert Y-up to Z-up
    # Quaternion for 90° rotation around X: [cos(45°), sin(45°), 0, 0] =
    # [0.707, 0.707, 0, 0]
    quat = Gf.Quatf(0.707, Gf.Vec3f(0.707, 0.0, 0.0))  # 90° around X-axis
    orient_op = xform_prim.AddOrientOp()
    orient_op.Set(quat)

    # ========================================================================
    # Pipeline Step 4: Apply Rigid Body and Collision Properties
    # ========================================================================
    # Add rigid body properties to the mesh (after mesh is set up)
    if xform_prim:
        xform_prim_obj = xform_prim.GetPrim()
        # Try to use Physics schema if available (USD Physics extension)
        try:
            from pxr import UsdPhysics
            # Apply RigidBodyAPI to enable rigid body physics
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(xform_prim_obj)
            if rigid_body_api:
                # Enable rigid body
                rigid_body_api.CreateRigidBodyEnabledAttr().Set(True)
        except (ImportError, AttributeError) as e:
            # USD Physics extension not available or API different,
            # use custom attributes
            # Add custom attributes that physics engines can interpret
            UsdPhysics.CollisionAPI.Apply(xform_prim_obj)

    # Add collision shape to the mesh
    if mesh_prim:
        mesh_prim_obj = mesh_prim.GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh_prim_obj)
        mesh_col_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim_obj)
        if mesh_col_api:
            approx_attr = mesh_col_api.GetApproximationAttr()
            approx_attr.Set(UsdPhysics.Tokens.convexDecomposition)
        # Add mass to the mesh (automatically calculated by physics engine)
        UsdPhysics.MassAPI.Apply(mesh_prim_obj)
    # Save the modified USD file
    stage.GetRootLayer().Export(usd_path)
    print(f"    Saved modified USD file to: {usd_path}")


def main(image_dir: str, output_dir: str):
    # 1. Setup paths and load model
    TAG = "hf"
    config_path = f"checkpoints/{TAG}/pipeline.yaml"
    
    print("Loading model...")
    inference = Inference(config_path, compile=False)
    print("Model loaded successfully!")
    
    # 2. Load input image and masks
    # Expect SAM-3D-Objects-style folder:
    #   image_dir/
    #     l/ (or m/ or s/)
    #        image.png, 0.png, 1.png, ...
    IMAGE_PATH = os.path.join(image_dir, "image.png")
    MASK_FOLDER = image_dir
    
    print(f"\nLoading image: {IMAGE_PATH}")
    image = load_image(IMAGE_PATH)
    print(f"Loading masks from: {MASK_FOLDER}")
    masks = load_masks(MASK_FOLDER, extension=".png")
    print(f"Found {len(masks)} masks")
    
    # Optional: display image with masks
    # display_image(image, masks)
    
    # 3. Generate Gaussian Splats (and per-object USDs) for each mask
    print(f"\nGenerating 3D reconstructions for {len(masks)} objects...")
    outputs = []
    # Create output directory: output_dir/image_dir_name/
    per_object_usd_dir = output_dir
    os.makedirs(per_object_usd_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        print(f"  Processing mask {i+1}/{len(masks)}...")
        usd_path = os.path.join(per_object_usd_dir, f"{i}.usd")
        output = inference(
            image,
            mask,
            seed=42,
            export_usd_path=usd_path,
        )

        if output.get("usd_path"):
            print(f"    USD saved for mask {i} at: {output['usd_path']}")
        else:
            print(f"    USD export failed for mask {i}; see logs for details.")

        outputs.append(output)

    print("All reconstructions complete!")
    
    # # 4. Combine all objects into a scene
    # print("\nCombining objects into scene...")
    # scene_gs = make_scene(*outputs)
    # scene_gs = ready_gaussian_for_video_rendering(scene_gs)
    
    # # Create output directory (include folder name of image)
    # output_dir = os.path.join("gaussians", "multi", IMAGE_NAME)
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Export Gaussian splat as point cloud
    # ply_path = f"{output_dir}/{IMAGE_NAME}.ply"
    # print(f"Saving scene to: {ply_path}")
    # scene_gs.save_ply(ply_path)
    # print("Scene saved!")
    
    # # 5. Render video and save as GIF
    # print("\nRendering video...")
    # video = render_video(
    #     scene_gs,
    #     r=1,
    #     fov=60,
    #     resolution=512,
    # )["color"]
    
    # gif_path = os.path.join(output_dir, f"{IMAGE_NAME}.gif")
    # print(f"Saving GIF to: {gif_path}")
    # imageio.mimsave(
    #     gif_path,
    #     video,
    #     format="GIF",
    #     duration=1000 / 30,  # 30fps
    #     loop=0,  # Loop indefinitely
    # )
    # print(f"GIF saved to: {gif_path}")
    
    # # 6. Optional: Launch interactive visualizer
    # print(f"\nScene reconstruction complete!")
    # print(f"  - Point cloud: {ply_path}")
    # print(f"  - Animation: {gif_path}")
    # print(f"\nTo view interactively, run:")
    # print(f"  interactive_visualizer('{ply_path}')")

    # 7. Apply transforms from scene_gs to each USD file
    print("\nApplying scene transforms to USD files...")
    for i, output in enumerate(outputs):
        if output.get("usd_path"):
            scale = output.get("scale", None)
            
            if scale is not None:
                print(f"  Applying scene transforms to USD {i}...")
                apply_transforms_to_usd(
                    output["usd_path"],
                    scale,
                    usd_scale_factor=1.0,  # Match the default used in inference
                    index=i,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Multi-object 3D reconstruction from "
            "SAM-3D-Objects-style folder."
        )
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help=(
            "Path to folder containing image.png and mask PNGs "
            "(0.png, 1.png, ...)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument(
        "--segment_mode",
        type=str,
        default="l",
        help="choose from 'l', 'm', 's'",
    )
    parser.add_argument(
        "--sam_ckpt_path",
        type=str,
        default="checkpoints/samv1/sam_vit_h_4b8939.pth",
        help="Path to SAM V1 checkpoint file.",
    )
    args = parser.parse_args()

    preprocess_input_dir = args.image_dir
    preprocess_output_dir = os.path.join(args.output_dir, "images")

    image_name = os.path.basename(os.path.normpath(preprocess_input_dir))
    sam3d_input_dir = os.path.join(preprocess_output_dir, image_name, args.segment_mode)
    sam3d_output_dir = os.path.join(args.output_dir, "usds", image_name)
    print(f"Preprocessing images from {preprocess_input_dir} to {preprocess_output_dir}")
    print(f"Processing USDs from {sam3d_input_dir} to {sam3d_output_dir}")
    preprocess(preprocess_input_dir, preprocess_output_dir, args.sam_ckpt_path)
    main(sam3d_input_dir, sam3d_output_dir)

