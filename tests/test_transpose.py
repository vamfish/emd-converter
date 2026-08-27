"""缓存分块转置的正确性测试。"""
import numpy as np
import pytest

from velox_file_analyzer2 import _transpose_hwc_to_fwc, _transpose_fwc_to_hwc

rng = np.random.default_rng(7)


@pytest.mark.parametrize('shape', [(17, 14, 5), (32, 64, 8), (64, 64, 64)])
def test_hwc_fwc_roundtrip(shape):
    h, w, f = shape
    arr = rng.integers(0, 1000, size=shape).astype(np.int16)
    fwc = _transpose_hwc_to_fwc(arr)
    assert fwc.shape == (f, h, w)
    assert fwc.flags.c_contiguous
    np.testing.assert_array_equal(fwc, arr.transpose(2, 0, 1))

    hwc = _transpose_fwc_to_hwc(fwc)
    assert hwc.shape == (h, w, f)
    assert hwc.flags.c_contiguous
    np.testing.assert_array_equal(hwc, arr)


def test_single_frame():
    arr = rng.integers(0, 100, size=(8, 9, 1)).astype(np.int16)
    fwc = _transpose_hwc_to_fwc(arr)
    np.testing.assert_array_equal(fwc[0], arr[:, :, 0])
    hwc = _transpose_fwc_to_hwc(fwc)
    np.testing.assert_array_equal(hwc, arr)