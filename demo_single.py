import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.unet import RestoreNet


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="path to degraded .npy image")
    p.add_argument("--gt", type=str, default=None, help="optional path to ground truth .npy image")
    p.add_argument("--ckpt", type=str, default="model_checkpoint.pth")
    p.add_argument("--output", type=str, default="demo_result.png")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RestoreNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    degraded = np.load(args.input).astype(np.float32)
    degraded_clipped = np.clip(degraded, 0.0, None)

    x = torch.from_numpy(degraded_clipped).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).clamp(0, 1).squeeze().cpu().numpy()

    n_panels = 3 if args.gt else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(np.clip(degraded, 0, 1), cmap="gray")
    axes[0].set_title("Degraded Input")
    axes[0].axis("off")

    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title("Restored Output (Model Prediction)")
    axes[1].axis("off")

    if args.gt:
        gt = np.load(args.gt).astype(np.float32)
        axes[2].imshow(np.clip(gt, 0, 1), cmap="gray")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
