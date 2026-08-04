"""
Generates side-by-side comparison images (degraded input -> restored output
-> ground truth) as PNG files, for the PPT "Results" slide visual evidence.

Usage:
    python visualize_results.py --degraded_dir /path/to/NoisyLR --pred_dir ./outputs --gt_dir /path/to/GT --save_dir ./comparisons --num_samples 6
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--degraded_dir", type=str, required=True)
    p.add_argument("--pred_dir", type=str, required=True)
    p.add_argument("--gt_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./comparisons")
    p.add_argument("--num_samples", type=int, default=6)
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.pred_dir) if f.endswith(".npy"))
    files = files[: args.num_samples]

    for fname in files:
        degraded = np.load(os.path.join(args.degraded_dir, fname)).astype(np.float32)
        pred = np.load(os.path.join(args.pred_dir, fname)).astype(np.float32)
        gt = np.load(os.path.join(args.gt_dir, fname)).astype(np.float32)

        degraded_disp = np.clip(degraded, 0, 1)
        pred_disp = np.clip(pred, 0, 1)
        gt_disp = np.clip(gt, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(degraded_disp, cmap="gray")
        axes[0].set_title("Degraded Input")
        axes[0].axis("off")

        axes[1].imshow(pred_disp, cmap="gray")
        axes[1].set_title("Restored Output")
        axes[1].axis("off")

        axes[2].imshow(gt_disp, cmap="gray")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = os.path.join(args.save_dir, fname.replace(".npy", ".png"))
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"saved {out_path}")

    print(f"\nDone. {len(files)} comparison images saved to {args.save_dir}")


if __name__ == "__main__":
    main()
