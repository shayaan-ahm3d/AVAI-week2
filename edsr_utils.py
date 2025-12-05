import random

from torch import Tensor
from torchvision.transforms.functional import crop
from torch.utils.tensorboard import SummaryWriter

def get_random_patch(low_res: Tensor, high_res: Tensor, patch_size: int, scale: int = 8) -> tuple[Tensor, Tensor]:
    _, low_res_height, low_res_width = low_res.shape
    
    # Calculate random crop position
    tx: int = random.randrange(0, low_res_width - patch_size + 1)
    ty: int = random.randrange(0, low_res_height - patch_size + 1)
    
    low_patch: Tensor = crop(low_res, ty, tx, patch_size, patch_size)
    
    # Crop HR (coordinates scaled)
    tx_hr, ty_hr = tx * scale, ty * scale
    high_patch_size: int = patch_size * scale
    high_patch: Tensor = crop(high_res, ty_hr, tx_hr, high_patch_size, high_patch_size)
    
    return low_patch, high_patch


def log_metrics(logger: SummaryWriter, mode: str, loss: float, psnr: float, ssim: float, lpips: float, step: int) -> None:
    logger.add_scalars(
            main_tag="loss",
            tag_scalar_dict={mode: loss},
            global_step=step
    )
    logger.add_scalars(
            main_tag="psnr",
            tag_scalar_dict={mode: psnr},
            global_step=step
    )
    logger.add_scalars(
            main_tag="ssim",
            tag_scalar_dict={mode: ssim},
            global_step=step
    )
    logger.add_scalars(
            main_tag="lpips",
            tag_scalar_dict={mode: lpips},
            global_step=step
    )