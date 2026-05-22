import argparse
import os
import sys

import numpy as np
import trimesh

sys.path.append("notebook")
from inference import Inference, load_image, load_mask


def build_guidance(
    mask_path,
    ss_guidance_scale,
    pose_guidance_scale,
    depth_guidance_scale,
    normal_guidance_scale,
    depth_map,
    steps_prefix,
    w_centroid,
    w_size,
):
    any_guidance = (
        (ss_guidance_scale is not None and ss_guidance_scale > 0)
        or (pose_guidance_scale is not None and pose_guidance_scale > 0)
        or (depth_guidance_scale is not None and depth_guidance_scale > 0)
        or (normal_guidance_scale is not None and normal_guidance_scale > 0)
    )
    if not any_guidance:
        return None

    from guidance import CompositeGuidance, DepthGuidance, PoseGuidance, ShapeGuidance, NormalGuidance

    debug_root = os.path.join("outputs", steps_prefix or "ss_steps", "guidance_debug")
    modules = []

    if ss_guidance_scale is not None and ss_guidance_scale > 0:
        print(f"[GUIDANCE] ShapeGuidance  scale={ss_guidance_scale}")
        modules.append(
            ShapeGuidance(
                mask_path=mask_path,
                shape_scale=ss_guidance_scale,
                device="cpu",
                debug_dir=os.path.join(debug_root, "shape"),
            )
        )

    if pose_guidance_scale is not None and pose_guidance_scale > 0:
        print(f"[GUIDANCE] PoseGuidance   scale={pose_guidance_scale}")
        modules.append(
            PoseGuidance(
                mask_path=mask_path,
                pose_scale=pose_guidance_scale,
                w_centroid=w_centroid,
                w_size=w_size,
                device="cpu",
            )
        )

    if depth_guidance_scale is not None and depth_guidance_scale > 0:
        if depth_map is None:
            print(
                "[GUIDANCE] WARNING: --depth-guidance-scale set but --depth-map not provided — skipping DepthGuidance"
            )
        else:
            print(
                f"[GUIDANCE] DepthGuidance  scale={depth_guidance_scale}  depth={depth_map}"
            )
            modules.append(
                DepthGuidance(
                    depth_path=depth_map,
                    mask_path=mask_path,
                    depth_scale=depth_guidance_scale,
                    device="cpu",
                )
            )

    if normal_guidance_scale is not None and normal_guidance_scale > 0:
        if depth_map is None:
            print(
                "[GUIDANCE] WARNING: --normal-guidance-scale set but --depth-map not provided — skipping NormalGuidance"
            )
        else:
            print(
                f"[GUIDANCE] NormalGuidance  scale={normal_guidance_scale}  depth={depth_map}"
            )
            modules.append(
                NormalGuidance(
                    depth_path=depth_map,
                    mask_path=mask_path,
                    normal_scale=normal_guidance_scale,
                    device="cpu",
                )
            )

    return CompositeGuidance(modules) if modules else None


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3D inference on Open3DHOI dataset"
    )
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--mask", required=True, help="Path to object mask")
    parser.add_argument(
        "--prefix", required=True, help="Prefix for saved steps e.g. occlusion_wrench"
    )
    parser.add_argument("--tag", default="hf", help="Checkpoint tag (default: hf)")
    parser.add_argument("--seed", type=int, default=42)

    # Guidance
    parser.add_argument(
        "--ss-guidance-scale", type=float, default=None, help="Shape guidance scale."
    )
    parser.add_argument(
        "--pose-guidance-scale", type=float, default=None, help="Pose guidance scale."
    )
    parser.add_argument(
        "--depth-guidance-scale",
        type=float,
        default=None,
        help="Depth guidance scale. Requires --depth-map.",
    )
    parser.add_argument(
        "--normal-guidance-scale",
        type=float,
        default=None,
        help="Normal guidance scale. Requires --depth-map."
    )
    parser.add_argument(
        "--depth-map",
        type=str,
        default=None,
        help="Path to GT depth .npy file (Open3DHOI format).",
    )
    parser.add_argument(
        "--w-centroid",
        type=float,
        default=1.0,
        help="[PoseGuidance] Weight for centroid loss.",
    )
    parser.add_argument(
        "--w-size", type=float, default=1.0, help="[PoseGuidance] Weight for size loss."
    )

    args = parser.parse_args()

    config_path = f"checkpoints/{args.tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    image = load_image(args.image)
    mask = load_mask(args.mask)

    guidance = build_guidance(
        mask_path=args.mask,
        ss_guidance_scale=args.ss_guidance_scale,
        pose_guidance_scale=args.pose_guidance_scale,
        depth_guidance_scale=args.depth_guidance_scale,
        normal_guidance_scale=args.normal_guidance_scale,
        depth_map=args.depth_map,
        steps_prefix=args.prefix,
        w_centroid=args.w_centroid,
        w_size=args.w_size,
    )

    output = inference(
        image,
        mask,
        seed=args.seed,
        steps_prefix=args.prefix,
        guidance=guidance,
    )

    out_dir = os.path.join("outputs", args.prefix)
    os.makedirs(out_dir, exist_ok=True)

    if "gs" in output:
        np.save(
            os.path.join(out_dir, "pred_points.npy"),
            output["gs"].get_xyz.detach().cpu().numpy(),
        )
        output["gs"].save_ply(os.path.join(out_dir, "gaussian.ply"))
        print(f"Saved pred_points.npy and gaussian.ply to {out_dir}/")

    if "mesh" in output:
        mesh = output["mesh"][0]
        if mesh.success:
            tm = trimesh.Trimesh(
                vertices=mesh.vertices.cpu().numpy(),
                faces=mesh.faces.cpu().numpy(),
            )
            tm.export(os.path.join(out_dir, "pred_mesh.obj"))
            print(f"Saved pred_mesh.obj to {out_dir}/")

    print(f"Done! Steps saved with prefix: {args.prefix}")


if __name__ == "__main__":
    main()
