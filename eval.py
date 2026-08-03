"""
Standalone evaluation script — required submission format.
Usage:
    python eval.py --test_dir /path/to/test/NoisyLR --output_dir /path/to/output --ckpt model_checkpoint.pth
"""
import argparse
import os
import numpy as np
import torch

from models.unet import RestoreNet


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", type=str, required=True,
                    help="path to folder with degraded .npy test images")
    p.add_argument("--output_dir", type=str, required=True,
                    help="path to save restored .npy outputs")
    p.add_argument("--ckpt", type=str, default="model_checkpoint.pth")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RestoreNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    files = sorted(f for f in os.listdir(args.test_dir) if f.endswith(".npy"))
    with torch.no_grad():
        for fname in files:
            arr = np.load(os.path.join(args.test_dir, fname)).astype(np.float32)
            arr = np.clip(arr, 0.0, None)
            x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(x).clamp(0, 1)
            pred = pred.squeeze().cpu().numpy()
            np.save(os.path.join(args.output_dir, fname), pred)

    print(f"Done. Restored {len(files)} images -> {args.output_dir}")


if __name__ == "__main__":
    main()
