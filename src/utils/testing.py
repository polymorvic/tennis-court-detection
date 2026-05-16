from datetime import datetime
from pathlib import Path

from src.schemas.testing import TestType


def build_output_dir(parent_dir: str | Path, test_type: TestType) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(parent_dir) / test_type.value / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir