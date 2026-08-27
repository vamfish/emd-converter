"""损坏文件与异常路径容错（M1）。"""
import numpy as np
import pytest

from velox_file_analyzer2 import VeloxFileAnalyzer
from tests import synthetic_emd as s


def test_corrupt_truncated_file_raises_cleanly(tmp_path):
    """截断的 EMD：构造失败而非崩溃/挂起。"""
    good = s.build_emd(tmp_path / 'good.emd', features=['camera'], shape=(8, 8))
    raw = good.read_bytes()
    bad = tmp_path / 'bad.emd'
    bad.write_bytes(raw[: len(raw) // 3])
    with pytest.raises(Exception) as ei:
        a = VeloxFileAnalyzer(str(bad))
        a.f.close()
    assert ei.value is not None


def test_missing_file(tmp_path):
    with pytest.raises(Exception):
        VeloxFileAnalyzer(str(tmp_path / 'nope.emd'))


def test_batch_continues_after_failure(tmp_path):
    """批量循环：坏文件后好文件仍能解析（模拟 GUI 行为）。"""
    good = s.build_emd(tmp_path / 'good.emd', features=['stem'], shape=(8, 8))
    bad = tmp_path / 'bad.emd'
    bad.write_bytes(b'this is not an hdf5 file at all........')

    analyzer = None
    failed_ok = False
    for fn in (bad, good):
        try:
            a = VeloxFileAnalyzer(str(fn))
            if analyzer is None:
                analyzer = a
        except Exception:
            failed_ok = True
    assert failed_ok
    assert analyzer is not None
    assert hasattr(analyzer, 'stem_data')
    analyzer.f.close()