"""损坏文件与异常路径容错（M1）。"""
import numpy as np
import pytest

from velox_file_analyzer2 import VeloxFileAnalyzer
from tests import synthetic_emd as s


def test_empty_crop_feature_skipped(tmp_path):
    """"空裁剪"记录（CropFeature 无 imageDisplay）：整文件仍可解析导出，
    仅跳过裁剪特征（复现 cichou_20260203_1051 0002 的 KeyError: 'imageDisplay'）。"""
    from emd_converter_gui import process_one_file, default_export_options
    f = s.build_emd(tmp_path / 'emptycrop.emd', features=['camera', 'crop_empty'],
                    shape=(16, 16), frames=1)
    a = VeloxFileAnalyzer(str(f))
    try:
        assert not hasattr(a, 'crop_data')          # 裁剪被优雅跳过
        assert hasattr(a, 'tem_data')               # 其余特征正常
    finally:
        a.f.close()
    out = tmp_path / 'out'
    stem, err = process_one_file((str(f), str(out), default_export_options()))
    assert err is None
    files = [p.name for p in (out / stem).iterdir()]
    assert files and not any('Crop' in n for n in files)  # 不产出裁剪文件


def test_partial_file_recovery_export(tmp_path):
    """缺 /Features 的半成品文件：解析不再崩溃，/Data/Image 数据按恢复模式导出。"""
    import h5py
    from emd_converter_gui import process_one_file, default_export_options

    f = s.build_partial_emd(tmp_path / 'part.emd', shape=(16, 16), frames=1)
    a = VeloxFileAnalyzer(str(f))
    try:
        assert len(a.recovered_images) == 1
        assert a.recovered_images[0]['image_name'] == 'part-Recovered'
        assert a.features == []
    finally:
        a.f.close()

    out = tmp_path / 'out'
    stem, err = process_one_file((str(f), str(out), default_export_options()))
    assert err is None
    names = sorted(p.name for p in (out / stem).iterdir())
    assert any('Recovered' in n for n in names), names
    with h5py.File(out / stem / 'part-Recovered.dm5', 'r') as hf:
        stored = hf['ImageList/[1]/ImageData/Data'][()]
    raw = s.default_image((16, 16), 1)
    np.testing.assert_array_equal(stored, raw.transpose(2, 0, 1))


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