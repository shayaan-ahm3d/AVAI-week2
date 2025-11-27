from enum import Enum
from pathlib import Path
from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class Mode(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


class Div2kDataset(Dataset):
    def __init__(self, low_res_dir: Path, high_res_dir: Path, mode: Mode) -> None:
        super().__init__()
        self.low_res = ImageFolder(root=low_res_dir)
        self.high_res = ImageFolder(root=high_res_dir)
        assert len(self.low_res) == len(self.high_res)
        self.mode = mode

    def __getitem__(self, index)-> tuple[Image, Image]:
        return self.low_res[index][0], self.high_res[index][0]
    
    def __len__(self) -> int:
        return len(self.low_res)