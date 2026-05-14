from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from examples.partial_grid_detection_analysis import main


if __name__ == "__main__":
    raise SystemExit(main(["--output-dir", "data/tutorials"]))
