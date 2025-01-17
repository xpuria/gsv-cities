from pathlib import Path

# Example default: adapt to your environment
BASE_DIR = Path(__file__).resolve().parent.parent  # gsv-cities/
GT_ROOT = str(BASE_DIR / "datasets" / "ground_truth" / "")  # e.g. "gsv-cities/datasets/ground_truth/"
SF_XS_PATH = str(BASE_DIR / "datasets" / "SF_XS" / "val" / "")  # e.g. "gsv-cities/datasets/SF_XS/val/"
TOKYO_XS_PATH = str(BASE_DIR / "datasets" / "Tokyo_XS" / "test" / "")  # e.g. "gsv-cities/datasets/Tokyo_XS/test/"