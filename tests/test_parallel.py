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


def test_compute_parallel_workers(tmp_path, monkeypatch):
    """内存自适应 worker 数：大文件自动降为串行。

    使用真实文件 + str 路径——与 GUI file_list 的实际元素类型一致
    （历史 bug：对 str 调 .stat() 使并行线程静默崩溃，mock 却掩盖了它）。
    """
    def make(name, size):
        p = tmp_path / name
        with open(p, 'wb') as fh:
            fh.truncate(size)   # 稀疏文件，瞬间完成
        return str(p)          # str！

    big = make('big.emd', 5 * 2**30)
    smalls = [make(f's{i}.emd', 64 * 2**20) for i in range(10)]

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
    # 不存在的路径不得抛异常（worker 决策容错）
    assert EMDConverterGUI._compute_parallel_workers(
        [str(tmp_path / 'ghost.emd')] + smalls) >= 1


def test_process_one_file_survives_gbk_console(tmp_path):
    """回归：spawn 子进程继承中文控制台编码 GBK(cp936)，导出日志含
    ✓/✗（U+2713/2717），GBK 无法编码——worker 必须自行切到 UTF-8，
    否则整批任务全部以 UnicodeEncodeError 失败（v0.3.1 exe 实测事故）。"""
    import io
    buf = io.BytesIO()
    gbk_out = io.TextIOWrapper(buf, encoding='gbk', errors='strict')
    f = s.build_emd(tmp_path / 'cam_gbk.emd', features=['camera'],
                    shape=(16, 16), frames=2)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = gbk_out
    try:
        stem, err = process_one_file((str(f), str(tmp_path / 'out'),
                                      default_export_options()))
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    assert err is None, f"GBK 控制台下任务失败: {err}"
    assert (tmp_path / 'out' / stem).exists()


def test_process_files_parallel_end_to_end(tmp_path):
    """端到端驱动真实 GUI 并行路径：str 文件列表 → 线程 + mainloop → 完成回调。

    这是能抓住"线程内 AttributeError 被静默吞掉、UI 永停
    '并行处理开启'"那类 bug 的集成测试（历史真实事故）。
    必须跑真实 mainloop——GUI 线程会调用 root.after()，主线程不在
    mainloop 中时 tkinter 抛 RuntimeError（行为与真实 GUI 不一致）。
    """
    import threading
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip('需要 tkinter')
    try:
        root = tk.Tk()
    except Exception:
        pytest.skip('需要可用显示环境')
    root.withdraw()
    old_stdout = sys.stdout
    try:
        import emd_converter_gui as G
        files = [s.build_emd(tmp_path / f'p{i}.emd', features=['camera'],
                             shape=(16, 16), frames=2) for i in range(3)]
        app = G.EMDConverterGUI(root)
        app.file_list = [str(f) for f in files]   # 关键：str，与 GUI 真实输入一致
        app.is_processing = True
        state = {}

        def fake_finished(success_count=0, failed=None):
            state['done'] = (success_count, list(failed or []))
            root.quit()
        app._conversion_finished = fake_finished
        root.after(150000, lambda: (state.setdefault('timeout', True), root.quit()))

        out = tmp_path / 'par_out'
        threading.Thread(target=app._process_files_parallel,
                         args=(out, G.default_export_options()),
                         daemon=True).start()
        root.mainloop()

        assert not state.get('timeout'), "并行处理超时未完成（死锁/线程静默死亡）"
        assert 'done' in state, "_conversion_finished 未被调度"
        assert state['done'][0] == 3, f"成功数异常: {state['done']}"
        assert len(list(out.rglob('*.dm5'))) >= 3
    finally:
        sys.stdout = old_stdout
        root.destroy()