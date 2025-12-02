from dataset import Div2kDataset, Mode
from inr_utils import convert_pixel_value_range, laplace, divergence, gradient, ssim
from siren import Siren

from pathlib import Path

import torch
from torch.nn import Module, MSELoss
from torchvision.transforms import Compose, ToTensor, Normalize
from skimage.metrics import peak_signal_noise_ratio
from lpips import LPIPS
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mse = MSELoss()
# LPIPS model to calculate metrics
lpips_model = LPIPS().eval()
# Since the whole image is our dataset, this is just the number of gradient descent steps.
total_steps = 1_000
steps_til_summary = total_steps // 10

low_res_path = Path("dataset/DIV2K_train_LR_x8")
high_res_path = Path("dataset/DIV2K_train_HR")
transform = Compose([
    ToTensor(),
    Normalize(mean=torch.Tensor([0.5, 0.5, 0.5]), std=torch.Tensor([0.5, 0.5, 0.5]))
    ])
dataset = Div2kDataset(low_root=low_res_path, high_root=high_res_path, transform=transform, mode=Mode.TEST)

def train_model(model: Module, low_res_image: torch.Tensor, steps: int) -> None:
    model.train()
    input_coords, ground_truth_pixel_values = Div2kDataset.get_coordinate_to_pixel_value_mapping(low_res_image)
    input_coords = input_coords.to(DEVICE)
    ground_truth_pixel_values = ground_truth_pixel_values.to(DEVICE)

    for step in range(steps):
        model_output_pixel_values, _ = model(input_coords)

        out = model_output_pixel_values.reshape([low_res_image.shape[1], low_res_image.shape[2], 3]).permute([2, 0, 1])
        gt = ground_truth_pixel_values.reshape([low_res_image.shape[1], low_res_image.shape[2], 3]).permute([2, 0, 1])

        loss = mse(out, gt)
        
        if not step % steps_til_summary:
            print("Step %d: Total Loss %0.9f" % (step, loss))

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

@torch.no_grad()
def evaluate_model(model: Module, high_res_image: torch.Tensor, index: int) -> tuple[float, float, float, float]:
    model.eval()
    input_coords, ground_truth_pixel_values = Div2kDataset.get_coordinate_to_pixel_value_mapping(high_res_image)
    input_coords = input_coords.to(DEVICE)
    ground_truth_pixel_values = ground_truth_pixel_values.to(DEVICE)

    model_output_pixel_values, _ = model(input_coords)
    input_coords = input_coords.cpu()

    def unflatten(flat: torch.Tensor) -> torch.Tensor:
        return flat.view(high_res_image.shape[1], high_res_image.shape[2], 3)

    gt = unflatten(ground_truth_pixel_values)
    out = unflatten(model_output_pixel_values)

    # metrics
    # image tensors should already be [-1, 1] for LPIPS
    lpips_score = lpips_model(gt.permute(2, 0, 1).unsqueeze(0), out.permute(2, 0, 1).unsqueeze(0)).cpu().item()

    model_output_pixel_values = model_output_pixel_values.cpu()
    ground_truth_pixel_values = ground_truth_pixel_values.cpu()
    loss: float = mse(model_output_pixel_values, ground_truth_pixel_values).item()

    gt = gt.cpu().numpy()
    out = out.cpu().numpy()
    # convert to NumPy arrays for these metrics
    psnr: float = peak_signal_noise_ratio(gt, out)
    mean_ssim: float = ssim(gt, out, data_range=2.0)
    
    # save every 10 generated images
    if (index % 1) == 0:
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        axes[0].imshow(convert_pixel_value_range(gt))
        axes[0].set_title("Ground Truth", fontsize=20)
        axes[1].imshow(convert_pixel_value_range(out))
        axes[1].set_title("Model Output", fontsize=20)
        
        plt.savefig(f"outputs/INR/{index}-SISR")
        plt.close()

    del gt, out
    return loss, psnr, mean_ssim, lpips_score

if __name__ == "__main__":
    log_file: Path = Path("outputs/INR/inr.csv")
    log_file.write_text("image,loss,psnr,ssim,lpips\n")

    with log_file.open("a") as log:
        for i, (low, high) in enumerate(iter(dataset)):
            # SIREN super resolution model: (x, y) -> (R, G, B)
            super_resolve = Siren(in_features=2,
                                  out_features=3,
                                  hidden_features=256,
                                  hidden_layers=3,
                                  outermost_linear=True).to(DEVICE)
            optimiser = torch.optim.Adam(super_resolve.parameters() , lr=1e-5)

            low = low.to(DEVICE)
            train_model(super_resolve, low, total_steps)
            low = low.cpu()
            del low

            high = high.to(DEVICE)
            lpips_model = lpips_model.to(DEVICE)
            loss, psnr, mean_ssim, lpips_score = evaluate_model(super_resolve, high, i)
            high = high.cpu()
            lpips_model = lpips_model.cpu()
            del high

            log.write(f"{i},{loss:.4f},{psnr:.4f},{mean_ssim:.4f},{lpips_score:.4f}\n")
            # try to prevent out of memory error
            super_resolve = super_resolve.cpu()
            del super_resolve
            torch.cuda.empty_cache()