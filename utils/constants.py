"""Repository paths shared by training, data loaders, and analysis utilities."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_PARAMS_DIR = ROOT_DIR / "exp" / "default_params.json"
EXPERIMENTS_PATH = ROOT_DIR / "exp" / "configs"
RESULTS_PATH = OUTPUT_DIR / "results"
LOG_DIR = OUTPUT_DIR / "logs"
