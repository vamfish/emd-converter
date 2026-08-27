"""dm5_writer：写出的 DM5 结构可读回且数据一致。"""
import numpy as np
import h5py

from velox_file_analyzer2 import dm5_writer, _transpose_hwc_to_fwc

rng = np.random.default_rng(11)


def _write(tmp_path, data, parameters, signal_extra=None):
    out = tmp_path / 'out.dm5'
    signal = {'data': data, 'metadata': {'foo': 'bar'},
              'color': {'red': 1, 'green': 1, 'blue': 1},
              'display_range': [0, 1000], 'gamma': 1.0}
    if signal_extra:
        signal.update(signal_extra)
    dm5_writer(out, signal, parameters)
    return out


def _read(out):
    with h5py.File(out, 'r') as f:
        ds = f['ImageList/[1]/ImageData/Data']
        stored = ds[()]
        dims_attrs = dict(f['ImageList/[1]/ImageData/Dimensions'].attrs)
    return stored, dims_attrs


def test_uint16_data_roundtrip(tmp_path):
    data = rng.integers(0, 4096, size=(16, 16, 1)).astype(np.uint16)
    out = _write(tmp_path, data, {'pixelsize': 0.1, 'pixelunit': 'nm'})
    stored, dims = _read(out)
    np.testing.assert_array_equal(stored, data.transpose(2, 0, 1))
    assert stored.dtype == np.uint16
    # 维度标定
    assert int(dims['[0]']) == 16 and int(dims['[1]']) == 16 and int(dims['[2]']) == 1


def test_int16_series_roundtrip(tmp_path):
    data = rng.integers(-1000, 1000, size=(8, 9, 4)).astype(np.int16)
    out = _write(tmp_path, data, {'pixelsize': 0.5, 'pixelunit': 'nm'})
    stored, dims = _read(out)
    np.testing.assert_array_equal(stored, data.transpose(2, 0, 1))
    assert int(dims['[0]']) == 9 and int(dims['[1]']) == 8 and int(dims['[2]']) == 4


def test_float32_roundtrip(tmp_path):
    data = rng.random((8, 8, 2)).astype(np.float32)
    out = _write(tmp_path, data, {'pixelsize': 1.0, 'pixelunit': 'px'})
    stored, dims = _read(out)
    np.testing.assert_array_equal(stored, data.transpose(2, 0, 1))
    assert stored.dtype == np.float32


def test_metadata_missing_degrades(tmp_path):
    """缺少 pixelsize/pixelunit 时不崩溃（M1 健壮性）。"""
    data = rng.integers(0, 100, size=(8, 8, 1)).astype(np.uint16)
    out = _write(tmp_path, data, {})
    stored, _ = _read(out)
    np.testing.assert_array_equal(stored, data.transpose(2, 0, 1))


def test_thumbnail_present(tmp_path):
    data = rng.integers(0, 100, size=(16, 32, 1)).astype(np.uint16)
    out = _write(tmp_path, data, {'pixelsize': 1.0, 'pixelunit': 'px'})
    with h5py.File(out, 'r') as f:
        thumb = f['ImageList/[0]/ImageData/Data']
        assert thumb.shape[0] <= 384 and thumb.shape[1] <= 384