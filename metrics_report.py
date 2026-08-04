"""
Computes SSIM, PSNR, and LPIPS between your model's restored outputs and the
ground truth images. Use this to get the numbers for your PPT "Results" slide.

Usage:
    python metrics_report.py --data_root /path/to/train_data --ckpt model_checkpoint.pth
"""
import argparse
import time
import torch
from torch.utils.data import DataLoader

from models.unet import RestoreNet
from dataset import DegradedDataset
from utils import ssim, psnr, lpips_distance, _LPIPS_AVAILABLE


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                    help="folder containing GT/ and NoisyLR/ (use a held-out val split)")
    p.add_argument("--ckpt", type=str, default="model_checkpoint.pth")
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if not _LPIPS_AVAILABLE:
        print("NOTE: 'lpips' package not installed -> LPIPS will be skipped.")
        print("      Install with: pip install lpips")

    ds = DegradedDataset(args.data_root, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = RestoreNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    total_ssim, total_psnr, total_lpips, n_lpips = 0.0, 0.0, 0.0, 0
    total_time, n_images = 0.0, 0
    n_batches = 0

    with torch.no_grad():
        for lr_img, gt_img in loader:
            lr_img, gt_img = lr_img.to(device), gt_img.to(device)

            start = time.time()
            pred = model(lr_img).clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start

            total_time += elapsed
            n_images += lr_img.shape[0]

            total_ssim += ssim(pred, gt_img).item()
            total_psnr += psnr(pred, gt_img).item()

            lp = lpips_distance(pred, gt_img, device)
            if lp is not None:
                total_lpips += lp
                n_lpips += 1

            n_batches += 1

    print("\n===== Results (for PPT Slide 6) =====")
    print(f"SSIM:  {total_ssim / n_batches:.4f}")
    print(f"PSNR:  {total_psnr / n_batches:.2f} dB")
    if n_lpips:
        print(f"LPIPS: {total_lpips / n_lpips:.4f}")
    else:
        print("LPIPS: skipped (lpips not installed)")
    print(f"Avg inference time: {(total_time / n_images) * 1000:.2f} ms/image")


if __name__ == "__main__":
    main()
