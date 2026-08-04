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

## 1. Train

```bash
python train.py --data_root /path/to/data_root --epochs 30
```

Saves the best checkpoint (lowest val loss) to `model_checkpoint.pth`.

## 2. Run inference on the test set (standalone submission script)

```bash
python eval.py --test_dir /path/to/test/NoisyLR --output_dir ./outputs --ckpt model_checkpoint.pth
```

Restores every `.npy` in `test_dir`, saves outputs to `output_dir`, and prints
average + total inference time (needed since the hackathon benchmarks speed).

## 3. Compute metrics for the Results slide (SSIM / PSNR / LPIPS)

```bash
python metrics_report.py --data_root /path/to/val_data --ckpt model_checkpoint.pth
```

Point `--data_root` at a held-out validation split (must contain `GT/` and
`NoisyLR/`, since LPIPS/SSIM/PSNR need ground truth to compare against).
Prints SSIM, PSNR, LPIPS, and average inference time.

## 4. Generate before/after comparison images (visual evidence for the PPT)

```bash
python visualize_results.py --degraded_dir /path/to/NoisyLR --pred_dir ./outputs --gt_dir /path/to/GT --save_dir ./comparisons
```

Saves PNG files showing Degraded Input | Restored Output | Ground Truth
side by side.

## 5. Freeze the environment (4th submission component)

After training on Kaggle, run:
```bash
pip freeze > requirements.txt
```
and include that file in your submission.

## Notes

- Degraded images may have pixel values outside [0,1] due to speckle noise — this is expected, the model learns to handle it.
- Model always upsamples by 2x (matches the dataset's 512->256 / 256->128 downsampling).
- Training loss = L1 + 0.2 * SSIM loss (fast to train). LPIPS is used only for
  reporting metrics, not inside the training loss, to keep training speed reasonable.
