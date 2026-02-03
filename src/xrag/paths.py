from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT = "src"
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = SRC / "results"
CACHE_DIR = SRC / "preprocess_cache"
CONFIG_YAML = SRC / "config.yml"
SECRETS_ENV = SRC / "secrets.env"

if not SECRETS_ENV.exists():
    raise RuntimeError("secrets file '{SECRETS_ENV}' does not exist. Create it and set values according to secrets.env.TEMPLATE")

load_dotenv(SECRETS_ENV)

if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir()
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir()
