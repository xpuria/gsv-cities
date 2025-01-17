from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import sys

# Adjust this import path to match your repo structure
MAIN_PATH = Path(__file__).resolve().parent.parent.parent / "utils"
sys.path.append(str(MAIN_PATH))

try:
    from config import GT_ROOT, SF_XS_PATH  # type: ignore
except ImportError:
    raise ImportError("Cannot import GT_ROOT or SF_XS_PATH from config. "
                      "Please ensure config.py is present in 'utils' and properly set up.")

class SFXSDataset(Dataset):
    """
    This class loads images for the San Francisco (SF-XS) dataset, used for either
    validation or testing. We rely on config.py for SF_XS_PATH and GT_ROOT paths.

    The dataset has two possible 'subset_name' modes:
      1) 'sfxs_val'
      2) 'sfxs_test'

    Each mode loads:
      - A list of reference (database) images
      - A list of query images
      - A ground-truth array specifying matches
    """

    def __init__(self, subset_name="sfxs_val", transform=None):
        """
        Args:
            subset_name (str): Must be 'sfxs_val' or 'sfxs_test'.
            transform (callable): A function/transform that takes in a PIL image and returns a transformed version.
        """
        super().__init__()
        self.subset_name = subset_name.lower()
        if self.subset_name not in ["sfxs_val", "sfxs_test"]:
            raise ValueError("subset_name must be either 'sfxs_val' or 'sfxs_test'.")

        self.transform = transform

        # Verify that the dataset root exists and has database/queries subfolders
        self.dataset_dir = Path(SF_XS_PATH)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"SF_XS_PATH directory not found: {self.dataset_dir}")

        db_dir = self.dataset_dir / "database"
        q_dir = self.dataset_dir / "queries"
        if not db_dir.is_dir() or not q_dir.is_dir():
            raise RuntimeError(f"SF-XS dataset folder must contain 'database' and 'queries': {self.dataset_dir}")

        # Load .npy arrays: references, queries, and ground-truth
        db_path = Path(GT_ROOT) / "SF_XS" / f"{self.subset_name}_dbImages.npy"
        q_path = Path(GT_ROOT) / "SF_XS" / f"{self.subset_name}_qImages.npy"
        gt_path = Path(GT_ROOT) / "SF_XS" / f"{self.subset_name}_gtImages.npy"
        
        self.db_images = np.load(db_path)
        self.q_images = np.load(q_path)
        self.ground_truth = np.load(gt_path, allow_pickle=True)

        # Combine references + queries
        self.all_images = np.concatenate((self.db_images, self.q_images))
        self.num_references = len(self.db_images)
        self.num_queries = len(self.q_images)

    def __getitem__(self, idx):
        """
        For a given index, return the corresponding image (transformed if requested)
        and the index itself (which you'll need for recall@K matching logic).
        """
        # The first 'num_references' belong to the reference set, the rest to queries
        img_rel_path = self.all_images[idx]  # e.g. 'database/xxx.jpg' or 'queries/xxx.jpg'
        img_full_path = self.dataset_dir / img_rel_path

        # Open and transform
        with Image.open(img_full_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            if self.transform:
                pil_img = self.transform(pil_img)
        return pil_img, idx

    def __len__(self):
        return len(self.all_images)

    def __repr__(self):
        return (f"SFXSDataset(subset_name='{self.subset_name}', "
                f"num_refs={self.num_references}, num_queries={self.num_queries})")


# Quick test
if __name__ == "__main__":
    ds = SFXSDataset("sfxs_val")
    print(ds)
    print("Total images:", len(ds))
    print("Ground truth shape:", ds.ground_truth.shape)