from dataset import Div2kDataset, get_random_patch
from edsr_model import Edsr
from siren import Siren
from models import get_net
from utils.common_utils import get_noise
from utils.denoising_utils import get_params, optimize

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from lpips import LPIPS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = 8
PATCH_SIZE_LR = 24
DIP_ITERATIONS = 500
INR_STEPS = 1000
EDSR_CHECKPOINT = "logs/edsr_x8_psnr=26.93220069800875.pth"

def get_metrics(gt_np, pred_np, lpips_model):
    # gt_np, pred_np are (H, W, C) in [0, 1]
    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
    ssim_val = structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=2)
    
    # LPIPS expects (B, C, H, W) in [-1, 1]
    gt_tensor = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) * 2.0 - 1.0
    pred_tensor = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) * 2.0 - 1.0
    
    with torch.no_grad():
        lpips_val = lpips_model(gt_tensor, pred_tensor).item()
        
    return psnr, ssim_val, lpips_val

def run_bicubic(low_tensor, high_shape):
    # low_tensor: (C, H, W)
    # high_shape: (H, W)
    out = F.interpolate(low_tensor.unsqueeze(0), size=high_shape, mode='bicubic', align_corners=False).squeeze(0)
    return out.clamp(0, 1).cpu().permute(1, 2, 0).numpy()

def run_edsr(low_tensor, checkpoint_path):
    model = Edsr(scale=SCALE, n_resblocks=16, n_feats=64).to(DEVICE)
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"EDSR checkpoint {checkpoint_path} not found")
    
    model.eval()
    with torch.no_grad():
        out = model(low_tensor.unsqueeze(0).to(DEVICE)).squeeze(0)
    return out.clamp(0, 1).cpu().permute(1, 2, 0).numpy()

def run_dip(low_tensor, high_shape):
    # DIP Setup
    input_depth = 3
    pad = 'reflection'
    net = get_net(
        input_depth, 'skip', pad,
        skip_n33d=128, skip_n33u=128, skip_n11=4,
        num_scales=5, upsample_mode='bilinear'
    ).to(DEVICE)
    
    net_input = get_noise(input_depth, 'noise', high_shape).to(DEVICE).detach()
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    
    low_tensor = low_tensor.to(DEVICE)
    mse = nn.MSELoss()
    
    reg_noise_std = 1.0 / 30.0
    exp_weight = 0.99
    
    out_avg = None
    
    def closure():
        nonlocal net_input, out_avg
        
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            
        out = net(net_input)
        
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1.0 - exp_weight)
            
        out_downsampled = F.interpolate(out, size=low_tensor.shape[-2:], mode='bicubic', align_corners=False)
        
        loss = mse(out_downsampled, low_tensor.unsqueeze(0))
        loss.backward()
        return loss
        
    params = get_params('net', net, net_input)
    optimize('adam', params, closure, 0.01, DIP_ITERATIONS)
    
    final_out = out_avg if out_avg is not None else net(net_input)
    return final_out.squeeze(0).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()

def run_inr(low_tensor, high_shape):
    low_tensor_norm = (low_tensor - 0.5) / 0.5
    
    model = Siren(in_features=2, out_features=3, hidden_features=256, hidden_layers=3, outermost_linear=True).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create coordinate grid for LR image
    _, h, w = low_tensor.shape
    coords_lr = Div2kDataset.get_coordinate_to_pixel_value_mapping(low_tensor_norm.unsqueeze(0))[0].to(DEVICE)
    pixels_lr = low_tensor_norm.permute(1, 2, 0).reshape(-1, 3).to(DEVICE)
    
    model.train()
    for _ in range(INR_STEPS):
        out_pixels, _ = model(coords_lr)
        loss = F.mse_loss(out_pixels, pixels_lr)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    # Inference on HR grid
    model.eval()
    # Create coordinate grid for HR image
    # We need a dummy HR tensor to generate coords
    dummy_hr = torch.zeros((3, high_shape[0], high_shape[1]))
    coords_hr = Div2kDataset.get_coordinate_to_pixel_value_mapping(dummy_hr.unsqueeze(0))[0].to(DEVICE)
    
    with torch.no_grad():
        out_pixels_hr, _ = model(coords_hr)
        
    out_hr = out_pixels_hr.reshape(high_shape[0], high_shape[1], 3).cpu().numpy()
    
    # Denormalize [-1, 1] -> [0, 1]
    out_hr = (out_hr * 0.5) + 0.5
    return np.clip(out_hr, 0, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Index of validation image to use")
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE_LR, help="LR patch size")
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    
    # Load Dataset
    val_low_path = Path("dataset/DIV2K_valid_LR_x8")
    val_high_path = Path("dataset/DIV2K_valid_HR")
    dataset = Div2kDataset(val_low_path, val_high_path)
    
    low, high = dataset[args.index]

    low_patch, high_patch = get_random_patch(low, high, args.patch_size, SCALE)
    
    print(f"LR Patch: {low_patch.shape}, HR Patch: {high_patch.shape}")
    
    gt_np = high_patch.permute(1, 2, 0).numpy()
    
    lpips_model = LPIPS().to(DEVICE).eval()
    
    results = {}
    metrics = {}
    
    print("Bicubic")
    results['Bicubic'] = run_bicubic(low_patch, high_patch.shape[1:])
    metrics['Bicubic'] = get_metrics(gt_np, results['Bicubic'], lpips_model)
    
    print("EDSR")
    results['EDSR'] = run_edsr(low_patch, EDSR_CHECKPOINT)
    metrics['EDSR'] = get_metrics(gt_np, results['EDSR'], lpips_model)
    
    print("DIP")
    results['DIP'] = run_dip(low_patch, high_patch.shape[1:])
    metrics['DIP'] = get_metrics(gt_np, results['DIP'], lpips_model)
    
    print("INR")
    results['INR'] = run_inr(low_patch, high_patch.shape[1:])
    metrics['INR'] = get_metrics(gt_np, results['INR'], lpips_model)
    
    # Plotting
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    # GT
    axes[0].imshow(gt_np)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    methods = ['Bicubic', 'EDSR', 'DIP', 'INR']
    for i, method in enumerate(methods):
        res = results[method]
        psnr, ssim_val, lpips_val = metrics[method]
        
        axes[i+1].imshow(res)
        axes[i+1].set_title(f"{method}\nPSNR: {psnr:.2f}\nSSIM: {ssim_val:.3f}\nLPIPS: {lpips_val:.3f}")
        axes[i+1].axis('off')
        
    plt.tight_layout()
    output_path = f"outputs/comparison_idx{args.index}.png"
    plt.savefig(output_path)
    print(f"Saved comparison to {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
