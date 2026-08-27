"""GUI 独立导出函数与并行调度测试（headless 安全）。"""
import sys
import types

import pytest

# CI/无头环境未必有 tkinter：GUI 模块导入需要它，这里用桩兜底
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

from emd_converter_gui import (  # noqa: E402
    process_one_file, default_export_options, export_by_type_standalone,
    EMDConverterGUI,
)
from tests import synthetic_emd as s  # noqa: E402


def test_process_one_file_dm5(tmp_path):
    """独立 worker 函数在合成 TEM 文件上产出 DM5。"""
    f = s.build_emd(tmp_path / 'cam.emd', features=['camera'], shape=(16, 16), frames=3)
    out = tmp_path / 'out'
    opts = default_export_options()
    opts.update({'tiff': False, 'png': False, 'csv': False})
    stem, err = process_one_file((str(f), str(out), opts))
    assert err is None
    dms = list((out / stem).glob('*.dm5'))
    assert len(dms) == 1 and dms[0].stat().st_size > 0


def test_process_one_file_reports_failure(tmp_path):
    """损坏文件返回错误信息而非抛出（批量不中断）。"""
    bad = tmp_path / 'bad.emd'
    bad.write_bytes(b'not an hdf5 file at all........')
    opts = default_export_options()
    stem, err = process_one_file((str(bad), str(tmp_path / 'out'), opts))
    assert err is not None and '无法打开' in err or err is not None


def test_options_respected(tmp_path):
    """dm5=False 时不产出 DM5。"""
    f = s.build_emd(tmp_path / 'stem.emd', features=['stem'], shape=(16, 16))
    out = tmp_path / 'out2'
    opts = default_export_options()
    opts.update({'dm5': False, 'tiff': False, 'png': False, 'csv': False})
    stem, err = process_one_file((str(f), str(out), opts))
    assert err is None
    assert list((out / stem).glob('*.dm5')) == []


def test_compute_parallel_workers(monkeypatch):
    """内存自适应 worker 数：大文件自动降为串行。"""
    big = types.SimpleNamespace(stat=lambda: types.SimpleNamespace(st_size=5 * 2**30))
    smalls = [types.SimpleNamespace(stat=lambda: types.SimpleNamespace(st_size=64 * 2**20))
              for _ in range(10)]

    holder = {'avail': 100 * 2**30}
    fake = types.ModuleType('psutil')
    fake.virtual_memory = lambda: types.SimpleNamespace(available=holder['avail'])
    monkeypatch.setitem(sys.modules, 'psutil', fake)

    # 充足内存 + 10 个小文件 -> 上限 8 且不超过文件数
    assert EMDConverterGUI._compute_parallel_workers(smalls) == 8
    # 最大文件本身就要 2.2×5GB=11GB < 100GB -> 可并行
    assert EMDConverterGUI._compute_parallel_workers(smalls + [big]) > 1
    # 内存吃紧（最大文件 2.2× 超可用内存）-> 自动串行
    holder['avail'] = 5 * 2**30
    assert EMDConverterGUI._compute_parallel_workers([big] + smalls) == 1

    # 无 psutil：保守假设 8GB
    monkeypatch.delitem(sys.modules, 'psutil')
    assert 1 <= EMDConverterGUI._compute_parallel_workers(smalls) <= 8