# Semicon Hackathon — AI-Based Restoration of Degraded Images

Joint speckle noise + gaussian noise removal + 2x super-resolution using a
residual U-Net style network with PixelShuffle upsampling.

## Setup

```bash
pip install -r requirements.txt
```

## Expected dataset layout

```
data_root/
    GT/          # ground truth .npy files (512x512 or 256x256)
    NoisyLR/     # degraded .npy files (256x256 or 128x128), same filenames as GT
```

## Train

```bash
python train.py --data_root /path/to/data_root --epochs 30
```

Saves best checkpoint to `model_checkpoint.pth`.

## Evaluate / Run inference (standalone script, no manual edits needed)

```bash
python eval.py --test_dir /path/to/test/NoisyLR --output_dir /path/to/output --ckpt model_checkpoint.pth
```

## Notes

- Degraded images may have pixel values outside [0,1] due to speckle noise — this is expected, the model learns to handle it.
- Model always upsamples by 2x (matches the dataset's 512->256 / 256->128 downsampling).
- Loss = L1 + 0.2 * SSIM loss. Add LPIPS (`pip install lpips`) for a stronger perceptual loss if time allows.
