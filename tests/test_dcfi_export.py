"""DCFI"仅导出最后一帧（积分图）"选项测试。

真实 DCFI 数据栈（IntegrationOperation 输出）的每一帧都是"截至第 k 帧的
累积积分图"，最后一帧才是全部叠加后的积分图像；选项开启时 DCFI 分支
只导出该末帧（2D），原始逐帧数据由 TEM/STEM 分支导出。
"""
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
import tifffile  # noqa: E402

from emd_converter_gui import (  # noqa: E402
    process_one_file, default_export_options,
)
from tests import synthetic_emd as s  # noqa: E402

SHAPE = (8, 8)
FRAMES = 4


def _dcfi_named(out_dir, stem):
    """DCFI 分支产物（文件名 = 显示标签 "DCFI(Ceta) of {stem}"）。"""
    files = list((out_dir / stem).glob('*DCFI*'))
    assert files, f"未找到 DCFI 导出文件: {sorted(p.name for p in (out_dir / stem).iterdir())}"
    return files


def _pick(files, suffixes):
    return next(p for p in files if p.suffix in suffixes)


def _dm5_data(path):
    with h5py.File(path, 'r') as hf:
        return hf['ImageList/[1]/ImageData/Data'][()]


def _export(tmp_path, name, last_frame_only):
    f = s.build_emd(tmp_path / f'{name}.emd', features=['dcfi'],
                    shape=SHAPE, frames=FRAMES)
    out = tmp_path / f'out_{name}'
    opts = default_export_options()
    opts['dcfi']['last_frame_only'] = last_frame_only
    stem, err = process_one_file((str(f), str(out), opts))
    assert err is None, f"导出失败: {err}"
    return out, stem


def test_option_on_exports_last_frame_only_dm5(tmp_path):
    """选项开（默认）：DM5 只含末帧，且内容 == dcfi_data[..., -1]。"""
    out, stem = _export(tmp_path, 'on', True)
    stored = _dm5_data(_pick(_dcfi_named(out, stem), ('.dm5',)))
    assert stored.shape == (1, *SHAPE)
    expect = s.default_image(SHAPE, FRAMES)[..., -1]  # (H, W) 最后一帧
    np.testing.assert_array_equal(stored[0], expect)


def test_option_on_tiff_is_2d_and_png_exists(tmp_path):
    """选项开：TIFF 为单帧 2D；PNG 正常产出。"""
    out, stem = _export(tmp_path, 'on2', True)
    files = _dcfi_named(out, stem)
    assert tifffile.imread(_pick(files, ('.tif', '.tiff'))).shape == SHAPE
    assert _pick(files, ('.png',)).stat().st_size > 0


def test_option_off_keeps_full_stack(tmp_path):
    """选项关：保持现状——导出完整 N 帧累积积分 stack。"""
    out, stem = _export(tmp_path, 'off', False)
    files = _dcfi_named(out, stem)
    full = s.default_image(SHAPE, FRAMES)
    stored = _dm5_data(_pick(files, ('.dm5',)))
    assert stored.shape == (FRAMES, *SHAPE)
    np.testing.assert_array_equal(stored, full.transpose(2, 0, 1))
    assert tifffile.imread(_pick(files, ('.tif', '.tiff'))).shape == (FRAMES, *SHAPE)


def test_single_frame_stack_untouched(tmp_path):
    """F=1 时选项开不崩溃，产物仍为 1 帧（行为与关一致）。"""
    f = s.build_emd(tmp_path / 'one.emd', features=['dcfi'], shape=SHAPE, frames=1)
    out = tmp_path / 'out_one'
    opts = default_export_options()  # 默认 last_frame_only=True
    stem, err = process_one_file((str(f), str(out), opts))
    assert err is None, f"导出失败: {err}"
    stored = _dm5_data(_pick(_dcfi_named(out, stem), ('.dm5',)))
    assert stored.shape == (1, *SHAPE)


def test_default_options_include_dcfi_group():
    """default_export_options 含 dcfi 组且默认开启（与 GUI 默认一致）。"""
    assert default_export_options()['dcfi'] == {'last_frame_only': True}
