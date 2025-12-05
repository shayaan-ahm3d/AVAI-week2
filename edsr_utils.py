from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

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

def get_unique_log_dir(log_dir: Path, scale: int, learning_rate: float, log_name: str = "edsr") -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'{log_name}_scale={scale}_lr={learning_rate:2.4f}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)