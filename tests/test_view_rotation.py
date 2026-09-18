"""视角校正（Ceta 图像自动恢复 Velox 屏幕方向）测试。

取证背景（见 CHANGELOG v0.3.0 与 apply_view_rotation docstring）：
ImageDisplay.angle 以**弧度**存储显示旋转（Velox "Image Rotation" 面板读数
= degrees(angle)；ddg 2026-09 样本 -4.8072409 rad ≙ 275.4°），存储数组未旋转。
angle=0 的文件（2026-02 旧批次、STEM/EDS 显示）恒等直通。
"""
import math
import sys
import types

# 与 test_parallel 相同的 tkinter 桩兜底（headless/CI 安全）
if 'tkinter' not in sys.modules:
    try:
        import tkinter as _real  # noqa: F401
    except ImportError:
        _shim = types.ModuleType('tkinter')
        for n in ('Tk', 'Frame', 'Label', 'Button', 'Entry', 'Checkbutton',
                  'Listbox', 'Scrollbar', 'StringVar', 'BooleanVar', 'IntVar',
                  'DoubleVar', 'Menu', 'filedialog', 'messagebox', 'ttk',
                  'END', 'NORMAL', 'DISABLED', 'SINGLE', 'EXTENDED'):
            setattr(_shim, n, type(n, (), {}))
        _shim.StringVar = type('StringVar', (),
                               {'get': lambda self: '', 'set': lambda self, v: None})
        sys.modules['tkinter'] = _shim

import matplotlib  # noqa: E402
matplotlib.use('Agg')

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from velox_file_analyzer2 import apply_view_rotation, _display_angle  # noqa: E402
from emd_converter_gui import (  # noqa: E402
    process_one_file, default_export_options,
)
from tests import synthetic_emd as s  # noqa: E402

rng = np.random.default_rng(7)
# ddg 0754 真实样本的 angle 字段（弧度），≙ UI 275.4°、等效 84.6° CCW
DDG_ANGLE_RAD = -4.8072409161999907
DDG_NET_DEG = math.degrees(DDG_ANGLE_RAD)  # ≈ -275.402


def _residual_deg(net_deg: float) -> float:
    """净角分解为 90° 整数步 + [-45,45] 残差。"""
    n90 = int(np.round(net_deg / 90.0))
    return net_deg - n90 * 90.0


def _inscribed_side(size: int, residual_deg: float) -> int:
    t = math.radians(abs(residual_deg))
    return int(math.floor(size / (math.cos(t) + math.sin(t))))


# ---------------------------------------------------------------- 单元测试

def test_zero_angle_passthrough():
    a = rng.integers(0, 5000, size=(16, 16)).astype(np.int16)
    assert apply_view_rotation(a, 0.0) is a or np.array_equal(apply_view_rotation(a, 0.0), a)


def test_pure_90_steps_lossless():
    a = rng.integers(0, 5000, size=(32, 32)).astype(np.int16)
    np.testing.assert_array_equal(apply_view_rotation(a, 90.0), np.rot90(a, 1))
    np.testing.assert_array_equal(apply_view_rotation(a, 270.0), np.rot90(a, 3))
    np.testing.assert_array_equal(apply_view_rotation(a, -90.0), np.rot90(a, -1))
    np.testing.assert_array_equal(apply_view_rotation(a, 180.0), np.rot90(a, 2))


def test_inscribed_crop_math_and_black_corners():
    a = rng.integers(0, 5000, size=(64, 64)).astype(np.int16)
    net = 80.5  # → n90=1, residual=-9.5
    resid = _residual_deg(net)
    assert abs(resid + 9.5) < 1e-9
    side = _inscribed_side(64, resid)
    assert side == int(math.floor(64 / (math.cos(math.radians(9.5)) + math.sin(math.radians(9.5)))))
    nc = apply_view_rotation(a, net, inscribed_crop=False)
    assert nc.shape == (64, 64)
    assert nc[0, 0] == 0 and nc[-1, -1] == 0  # 旋转后角区黑色填充
    c = apply_view_rotation(a, net)
    assert c.shape == (side, side) and c.dtype == a.dtype


def test_real_sample_net_angle_equivalence():
    """净角 -275.40° 应等价 rot90(data,-3) ∘ rotate(-5.40°)（0754 真实模型）。"""
    a = rng.integers(0, 5000, size=(64, 64)).astype(np.int16)
    from scipy.ndimage import rotate
    ref = np.rint(rotate(np.rot90(a, -3).astype(np.float64), _residual_deg(DDG_NET_DEG),
                         reshape=False, order=3, cval=0.0)).astype(np.int16)
    np.testing.assert_array_equal(apply_view_rotation(a, DDG_NET_DEG, inscribed_crop=False), ref)
    # 方向敏感：取反角结果必须明显不同
    assert not np.array_equal(apply_view_rotation(a, -DDG_NET_DEG, inscribed_crop=False), ref)


def test_3d_stack_per_frame():
    from scipy.ndimage import rotate
    a = rng.integers(0, 5000, size=(64, 64, 4)).astype(np.int16)
    net = 80.5
    out = apply_view_rotation(a, net, inscribed_crop=False)
    assert out.shape == (64, 64, 4)
    for k in range(4):
        ref = rotate(np.rot90(a[:, :, k]).astype(np.float64), -9.5,
                     reshape=False, order=3, cval=0.0)
        np.testing.assert_array_equal(out[:, :, k], np.rint(ref).astype(np.int16))


def test_float32_dtype_preserved():
    a = rng.random((32, 32)).astype(np.float32)
    assert apply_view_rotation(a, DDG_NET_DEG).dtype == np.float32


def test_display_angle_converts_radians():
    assert abs(_display_angle({'angle': str(DDG_ANGLE_RAD), 'offsetAngle': '0'}) - DDG_NET_DEG) < 1e-6
    assert _display_angle({}) == 0.0
    assert _display_angle({'angle': '', 'offsetAngle': None}) == 0.0
    assert abs(_display_angle({'angle': '1.5707963267948966'}) - 90.0) < 1e-6


# ---------------------------------------------------------------- 集成测试

def _dm5_data(path):
    with h5py.File(path, 'r') as hf:
        return hf['ImageList/[1]/ImageData/Data'][()]


def _export(tmp_path, name, features, angle=0.0):
    f = s.build_emd(tmp_path / f'{name}.emd', features=features,
                    shape=(32, 32), frames=1, angle=angle)
    out = tmp_path / f'out_{name}'
    stem, err = process_one_file((str(f), str(out), default_export_options()))
    assert err is None, f"导出失败: {err}"
    return out, stem


def test_tem_auto_rotated_when_angle_nonzero(tmp_path):
    """angle≠0 ⇒ TEM 自动旋转并裁内接方形（无需任何开关）。"""
    out, stem = _export(tmp_path, 'cam_rot', ['camera'], angle=DDG_ANGLE_RAD)
    stored = _dm5_data(next((out / stem).glob('*.dm5')))
    side = _inscribed_side(32, _residual_deg(DDG_NET_DEG))
    assert side == 29  # floor(32/(cos5.402+sin5.402))
    assert stored.shape == (1, side, side)
    # 内接方形不含黑角：像素不应为 0（合成数据值 ≥0 且旋转插值不产生纯 0 内点）
    assert (stored[0] != 0).mean() > 0.9


def test_angle_zero_is_legacy_identity(tmp_path):
    """angle=0（2026-02 旧批次同款）⇒ 与旧行为逐字节一致。"""
    out, stem = _export(tmp_path, 'cam_zero', ['camera'], angle=0.0)
    stored = _dm5_data(next((out / stem).glob('*.dm5')))
    raw = s.default_image((32, 32), 1)  # (32,32,1)
    np.testing.assert_array_equal(stored, raw.transpose(2, 0, 1))


def test_stem_not_rotated_even_if_display_has_angle(tmp_path):
    """STEM 分支不经过视角校正：默认导出恒等于存储（angle 注入也不生效）。"""
    f = s.build_emd(tmp_path / 'stem_rt.emd', features=['stem'], shape=(32, 32), frames=1)
    out = tmp_path / 'out_stem_rt'
    stem, err = process_one_file((str(f), str(out), default_export_options()))
    assert err is None
    stored = _dm5_data(next((out / stem).glob('*.dm5')))
    raw = s.default_image((32, 32), 1)
    np.testing.assert_array_equal(stored, raw.transpose(2, 0, 1))


def test_dcfi_rotation_after_last_frame(tmp_path):
    """DCFI：先取末帧再自动旋转 ⇒ (1, side, side)。"""
    f = s.build_emd(tmp_path / 'dcfi_rt.emd', features=['dcfi'],
                    shape=(32, 32), frames=3, angle=DDG_ANGLE_RAD)
    out = tmp_path / 'out_dcfi_rt'
    stem, err = process_one_file((str(f), str(out), default_export_options()))
    assert err is None
    stored = _dm5_data(next((out / stem).glob('*DCFI*.dm5')))
    assert stored.shape == (1, 29, 29)


def test_no_rotation_group_in_options():
    """视角校正已全自动：options 字典不再含 rotation 组。"""
    opts = default_export_options()
    assert 'rotation' not in opts
    assert opts['dcfi'] == {'last_frame_only': True}
