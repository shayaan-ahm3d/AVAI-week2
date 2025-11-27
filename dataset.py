from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


class Mode(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


class Div2kDataset(Dataset):
    """Paired DIV2K samples pulled from separate low/high directories."""

    def __init__(
        self,
        low_root: Path,
        high_root: Path,
        mode: Mode = Mode.TRAIN,
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        target_transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp")
    ) -> None:
        self.low_root = Path(low_root)
        self.high_root = Path(high_root)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = tuple(ext.lower() for ext in extensions)

        self.low_index = self._index_files(self.low_root)
        self.high_index = self._index_files(self.high_root)

        shared_keys = sorted(set(self.low_index.keys()) & set(self.high_index.keys()))
        if not shared_keys:
            raise ValueError(
                f"No matching filenames between {self.low_root} and {self.high_root}."
            )

        self.pairs: List[Tuple[Path, Path]] = [
            (self.low_index[key], self.high_index[key]) for key in shared_keys
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        low_path, high_path = self.pairs[idx]

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        if self.transform:
            low_img = self.transform(low_img)
        if self.target_transform:
            high_img = self.target_transform(high_img)

        return low_img, high_img

    def _index_files(self, root: Path) -> Dict[str, Path]:
        files = [
            path for path in Path(root).rglob("*")
            if path.is_file() and path.suffix.lower() in self.extensions
        ]
        index: Dict[str, Path] = {}
        for file in files:
            key = file.stem
            if key in index:
                raise ValueError(
                    f"Duplicate filename stem '{key}' found under {root}."
                )
            index[key] = file
        return index