import torch
import torch.nn.functional as F


def _gaussian_window(window_size, sigma):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _create_window(window_size=11, sigma=1.5):
    _1d = _gaussian_window(window_size, sigma).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
    return _2d


def ssim(img1, img2, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """img1, img2: (B,1,H,W) tensors in [0,1]"""
    window = _create_window(window_size).to(img1.device)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad)
    mu2 = F.conv2d(img2, window, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def ssim_loss(img1, img2):
    return 1 - ssim(img1, img2)


def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse.item() == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
