"""元数据健壮性：缺失像素尺寸/单位时的降级行为（M1）。"""
import numpy as np
import pytest

from velox_file_analyzer2 import VeloxFileAnalyzer


def test_pixel_size_degradation():
    """_get_pixel_size 对缺失/损坏元数据降级为 (1.0, 'px')。"""
    a = VeloxFileAnalyzer.__new__(VeloxFileAnalyzer)
    assert a._get_pixel_size({}) == (1.0, 'px')
    assert a._get_pixel_size({'BinaryResult': {}}) == (1.0, 'px')
    assert a._get_pixel_size({'BinaryResult': {'PixelSize': {}}}) == (1.0, 'px')


def test_pixel_size_units_conversion():
    a = VeloxFileAnalyzer.__new__(VeloxFileAnalyzer)
    meta = {'BinaryResult': {'PixelSize': {'width': '9.0e-11'},
                             'PixelUnitX': 'm'}}
    size, unit = a._get_pixel_size(meta)
    assert unit == 'nm' and size == pytest.approx(0.09)  # 0.09 nm

    meta2 = {'BinaryResult': {'PixelSize': {'width': '2.0'},
                              'PixelUnitX': '1/m'}}
    size2, unit2 = a._get_pixel_size(meta2)
    assert unit2 == '1/nm' and size2 == pytest.approx(2.0e-9)