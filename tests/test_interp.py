"""双线性插值：向量化实现 vs scipy RectBivariateSpline 数值对照。"""
import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from velox_file_analyzer2 import _vectorized_bilinear_interp

rng = np.random.default_rng(42)


@pytest.mark.parametrize('shape', [(16, 16), (37, 23), (100, 80)])
@pytest.mark.parametrize('dtype', [np.int16, np.float32, np.uint16])
def test_matches_scipy_linear_spline(shape, dtype):
    h, w = shape
    arr = rng.integers(0, 1000, size=shape).astype(dtype)
    # 随机采样点（全部在界内）
    n = 500
    xs = rng.uniform(0, w - 1, n)
    ys = rng.uniform(0, h - 1, n)

    spl = RectBivariateSpline(np.arange(w), np.arange(h), arr.T, kx=1, ky=1)
    # 逐点求值（spl(xs, ys) 是网格求值，不是逐点）
    expected = np.array([spl(x, y)[0, 0] for x, y in zip(xs, ys)])

    got = _vectorized_bilinear_interp(arr, xs, ys)

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_interpolates_within_range():
    arr = np.array([[0, 10], [20, 30]], dtype=np.int16)
    # 中心 (0.5, 0.5) = (0+10+20+30)/4 = 15
    assert _vectorized_bilinear_interp(arr, np.array([0.5]), np.array([0.5]))[0] == pytest.approx(15.0)
    # 角点
    assert _vectorized_bilinear_interp(arr, np.array([0.0]), np.array([0.0]))[0] == pytest.approx(0.0)
    assert _vectorized_bilinear_interp(arr, np.array([1.0]), np.array([1.0]))[0] == pytest.approx(30.0)


def test_out_of_bounds_clamped():
    arr = rng.integers(0, 100, size=(8, 8)).astype(np.int16)
    xs = np.array([-5.0])
    ys = np.array([-5.0])
    got = _vectorized_bilinear_interp(arr, xs, ys)[0]
    assert got == pytest.approx(float(arr[0, 0]))
    # 上界
    xs2 = np.array([1000.0])
    ys2 = np.array([1000.0])
    got2 = _vectorized_bilinear_interp(arr, xs2, ys2)[0]
    assert got2 == pytest.approx(float(arr[-1, -1]))