from pathlib import Path

# Project root = 3 levels up from this file
ROOT = Path(__file__).resolve().parent.parent.parent

DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"
NOTEBOOKS = ROOT / "notebooks"
SRC = ROOT / "src"
MODELS = SRC / "models"
