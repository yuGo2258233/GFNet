import os
import cv2
import json
import torch
import numpy as np
import argparse

from PIL import Image
import matplotlib.pyplot as plt
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches

from model.network import GFNet
from estimation import estimate_homography


def load_model(conf_path, ckpt_path, device):
    with open(conf_path, "r") as file:
        conf = json.load(file)
    training_resolution = (448, 448)
    upsampling_resolution = (560, 560)

    model = GFNet(
        conf=conf,
        initial_res=training_resolution,
        upsample_res=upsampling_resolution,
        symmetric=True,
        upsample_preds=True,
        attenuate_cert=True,
    ).to(device)

    states = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(states["model"])
    model.eval()
    return model


def visualize_matches(im_1, im_2, pos_a, pos_b, inliers_mask, save_path, img_shape=None):
    # Visualize matches
    plt.clf()
    # If inliers_mask is None (homography failed), show all as tentative or handle gracefully
    if inliers_mask is None:
        inliers_indices = []
        mask = np.zeros(len(pos_a), dtype=bool)  # All bad
    else:
        # inliers_mask from cv2.findHomography is (N, 1) uint8
        mask = inliers_mask.ravel().astype(bool)
        inliers_indices = np.where(mask)[0]

    # If no inliers, assume empty
    if len(inliers_indices) == 0:
        # Just save the images side by side?
        # Or just skip
        return

    viz_pos_a = pos_a[inliers_indices].copy()
    viz_pos_b = pos_b[inliers_indices].copy()

    if img_shape is not None:
        vis_w, vis_h = img_shape
        orig_w1, orig_h1 = im_1.size
        orig_w2, orig_h2 = im_2.size
        viz_pos_a[:, 0] *= vis_w / orig_w1
        viz_pos_a[:, 1] *= vis_h / orig_h1
        viz_pos_b[:, 0] *= vis_w / orig_w2
        viz_pos_b[:, 1] *= vis_h / orig_h2
        im_1 = im_1.resize((vis_w, vis_h), Image.BILINEAR)
        im_2 = im_2.resize((vis_w, vis_h), Image.BILINEAR)

    draw_LAF_matches(
        KF.laf_from_center_scale_ori(
            torch.from_numpy(viz_pos_a).view(1, -1, 2),
            torch.ones(viz_pos_a.shape[0]).view(1, -1, 1, 1),
            torch.ones(viz_pos_a.shape[0]).view(1, -1, 1),
        ),
        KF.laf_from_center_scale_ori(
            torch.from_numpy(viz_pos_b).view(1, -1, 2),
            torch.ones(viz_pos_b.shape[0]).view(1, -1, 1, 1),
            torch.ones(viz_pos_b.shape[0]).view(1, -1, 1),
        ),
        torch.arange(viz_pos_a.shape[0]).view(-1, 1).repeat(1, 2),
        np.array(im_1),
        np.array(im_2),
        [True] * len(viz_pos_a),  # All passed are inliers
        draw_dict={
            "inlier_color": (0.2, 1, 0.4),
            "tentative_color": (1, 0, 0),
            "feature_color": (0.2, 0.5, 1),
            "vertical": False,
        },
    )
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        default="Input/input.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--output_path", type=str, default="Output", help="Directory to save results"
    )
    parser.add_argument("--conf_path", type=str, default="configs/basic.json")
    parser.add_argument("--ckpt_path", type=str, default="ckpts/basic/latest.pth")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--skip_visualization", action="store_true", help="Skip saving match images"
    )
    parser.add_argument(
        "--vis_resolution",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize visualization images to this resolution (default: original resolution)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    matches_dir = os.path.join(args.output_path, "matches")
    vis_dir = os.path.join(args.output_path, "vis")

    os.makedirs(matches_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    model = load_model(args.conf_path, args.ckpt_path, args.device)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        return

    # Read all frames first? Video might be long.
    # Process pair by pair.

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Video has no frames")
        return

    prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    prev_im = Image.fromarray(prev_frame_rgb)

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {total_frames}")

    runtimes = []
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
        curr_im = Image.fromarray(curr_frame_rgb)

        # Estimate homography prev -> curr
        H, pos_a, pos_b, runtime, inliers_mask = estimate_homography(
            model, prev_im, curr_im
        )
        runtimes.append(runtime)

        # Save matches data per frame
        match_save_path = os.path.join(
            matches_dir, f"match_{frame_idx}_{frame_idx+1}.npy"
        )
        np.save(
            match_save_path,
            {"keypoint0": pos_a, "keypoint1": pos_b},
            allow_pickle=True,
        )

        # Visualize
        if not args.skip_visualization:
            viz_path = os.path.join(vis_dir, f"match_{frame_idx}_{frame_idx+1}.png")
            visualize_matches(
                prev_im, curr_im, pos_a, pos_b, inliers_mask, viz_path,
                img_shape=tuple(args.vis_resolution) if args.vis_resolution else None,
            )

        prev_im = curr_im
        print(
            f"Processing frame {frame_idx} -> {frame_idx + 1} / {total_frames}. Runtime: {runtime:.4f}s"
        )
        frame_idx += 1

    cap.release()

    np.save(os.path.join(args.output_path, "runtime.npy"), np.array(runtimes))

    print(f"Processed {frame_idx} pairs.")
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
