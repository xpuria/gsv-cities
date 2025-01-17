from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import sys

# Adjust this import path to match your repo structure
MAIN_PATH = Path(__file__).resolve().parent.parent.parent / "utils"
sys.path.append(str(MAIN_PATH))

try:
    from config import GT_ROOT, TOKYO_XS_PATH  # type: ignore
except ImportError:
    raise ImportError("Cannot import GT_ROOT or TOKYO_XS_PATH from config. "
                      "Please ensure config.py is correct.")

class TokyoXSDataset(Dataset):
    """
    Loads the Tokyo-XS dataset for testing.

    This dataset is structured similarly to SF-XS, but typically
    only has a single 'test' subset. The references and queries
    are loaded from .npy files, while images are loaded from a
    directory with 'database' and 'queries' subfolders.
    """
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

        self.dataset_dir = Path(TOKYO_XS_PATH)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Tokyo XS dataset path not found: {self.dataset_dir}")

        # Check database / queries subfolders
        db_folder = self.dataset_dir / "database"
        q_folder = self.dataset_dir / "queries"
        if not db_folder.is_dir() or not q_folder.is_dir():
            raise RuntimeError(f"Missing 'database' or 'queries' folder in {self.dataset_dir}")

        # Load references, queries, and ground-truth from .npy
        db_path = Path(GT_ROOT) / "Tokyo_XS" / "tokyoxs_test_dbImages.npy"
        q_path = Path(GT_ROOT) / "Tokyo_XS" / "tokyoxs_test_qImages.npy"
        gt_path = Path(GT_ROOT) / "Tokyo_XS" / "tokyoxs_test_gtImages.npy"

        self.db_images = np.load(db_path)
        self.q_images = np.load(q_path)
        self.ground_truth = np.load(gt_path, allow_pickle=True)

        # reference images come first, queries second
        self.all_images = np.concatenate((self.db_images, self.q_images))
        self.num_references = len(self.db_images)
        self.num_queries = len(self.q_images)

    def __getitem__(self, idx):
        """
        Return the image at 'idx' (from references if idx < num_references,
        otherwise from queries) plus the index for further use in evaluation.
        """
        relative_path = self.all_images[idx]
        full_path = self.dataset_dir / relative_path
        with Image.open(full_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            if self.transform:
                pil_img = self.transform(pil_img)
        return pil_img, idx

    def __len__(self):
        return len(self.all_images)

    def __repr__(self):
        return (f"TokyoXSDataset(test_only, "
                f"num_refs={self.num_references}, num_queries={self.num_queries})")


# Quick test
if __name__ == "__main__":
    ds = TokyoXSDataset()
    print(ds)
    print("Total images:", len(ds))
    print("Ground truth shape:", ds.ground_truth.shape)