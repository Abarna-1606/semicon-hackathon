import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from models.unet import RestoreNet
from dataset import DegradedDataset
from utils import ssim_loss, psnr


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/kaggle/input/semicon-dataset",
                    help="folder containing GT/ and NoisyLR/ subfolders")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ckpt", type=str, default="model_checkpoint.pth")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    full_ds = DegradedDataset(args.data_root, augment=True)
    val_len = max(1, int(0.1 * len(full_ds)))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = RestoreNet().to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    l1 = nn.L1Loss()

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for lr_img, gt_img in train_loader:
            lr_img, gt_img = lr_img.to(device), gt_img.to(device)
            opt.zero_grad()
            pred = model(lr_img)
            loss = l1(pred, gt_img) + 0.2 * ssim_loss(pred, gt_img)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        val_loss, val_psnr = 0.0, 0.0
        with torch.no_grad():
            for lr_img, gt_img in val_loader:
                lr_img, gt_img = lr_img.to(device), gt_img.to(device)
                pred = model(lr_img).clamp(0, 1)
                val_loss += l1(pred, gt_img).item()
                val_psnr += psnr(pred, gt_img).item()

        avg_train = total_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        avg_psnr = val_psnr / max(1, len(val_loader))
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss: {avg_train:.4f} "
              f"- val_loss: {avg_val:.4f} - val_psnr: {avg_psnr:.2f}dB")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), args.ckpt)
            print(f"  -> saved checkpoint ({args.ckpt})")


if __name__ == "__main__":
    main()
