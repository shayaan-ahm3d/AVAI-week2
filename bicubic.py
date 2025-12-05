import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from pathlib import Path
import csv
from skimage.metrics import peak_signal_noise_ratio
from lpips import LPIPS

from dataset import Div2kDataset, Mode
from inr_utils import ssim

SCALE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_CSV = Path("bicubic_x16.csv")

low_path = Path("dataset/DIV2K_valid_LR_x8")
high_path = Path("dataset/DIV2K_valid_HR")

def main():
    dataset = Div2kDataset(low_path, high_path, transform=ToTensor(), mode=Mode.TEST)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    lpips_model = LPIPS(net='alex').to(DEVICE)

    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image", "baseline_psnr", "baseline_ssim", "baseline_lpips"])

        print(f"Starting Bicubic Upscaling (x{SCALE}) on {len(dataset)} images...")
        
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0

        for i, (low, high) in enumerate(dataloader):
            low = low.to(DEVICE)
            high = high.to(DEVICE)

            # Downscale x8 input by 2x to simulate x16 input
            low_downscaled = F.interpolate(low, scale_factor=0.5, mode='bicubic', align_corners=False)

            # Bicubic Upscaling from x16 to HR
            # Upscale to the size of the HR image
            upscaled = F.interpolate(low_downscaled, size=high.shape[2:], mode='bicubic', align_corners=False)
            
            # Clamp to [0, 1]
            upscaled = upscaled.clamp(0, 1)

            # Calculate Metrics
            
            # PSNR & SSIM (Numpy, [0, 1], HWC)
            upscaled_np = upscaled.squeeze().cpu().numpy().transpose(1, 2, 0)
            high_np = high.squeeze().cpu().numpy().transpose(1, 2, 0)
            
            psnr_val = peak_signal_noise_ratio(high_np, upscaled_np, data_range=1.0)
            ssim_val = ssim(high_np, upscaled_np, data_range=1.0)
            
            # LPIPS (Tensor, [-1, 1], BCHW)
            upscaled_scaled = upscaled * 2.0 - 1.0
            high_scaled = high * 2.0 - 1.0
            lpips_val = lpips_model(high_scaled, upscaled_scaled).item()

            writer.writerow([i, psnr_val, ssim_val, lpips_val])
            
            total_psnr += psnr_val
            total_ssim += ssim_val
            total_lpips += lpips_val
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} images")

        avg_psnr = total_psnr / len(dataset)
        avg_ssim = total_ssim / len(dataset)
        avg_lpips = total_lpips / len(dataset)

        print(f"\nAverage Results:")
        print(f"PSNR: {avg_psnr:.4f}")
        print(f"SSIM: {avg_ssim:.4f}")
        print(f"LPIPS: {avg_lpips:.4f}")
        
        writer.writerow(["Average", avg_psnr, avg_ssim, avg_lpips])
        print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()