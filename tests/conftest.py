"""pytest 共享夹具。"""
import sys
from pathlib import Path

import pytest

# 确保项目根目录可导入 velox_file_analyzer2
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests import synthetic_emd  # noqa: E402


@pytest.fixture
def emd_builder(tmp_path):
    """返回 (build_emd 函数, tmp_path)。"""
    return synthetic_emd.build_emd, tmp_path


@pytest.fixture
def camera_file(emd_builder):
    build, d = emd_builder
    p = build(d / 'camera.emd', features=['camera'], shape=(16, 16), frames=1)
    return p


@pytest.fixture
def camera_series_file(emd_builder):
    build, d = emd_builder
    p = build(d / 'camera_series.emd', features=['camera'], shape=(16, 16), frames=5)
    return p