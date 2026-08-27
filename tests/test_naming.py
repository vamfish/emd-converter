"""命名与工具函数测试。"""
from pathlib import Path

from velox_file_analyzer2 import add_suffix_safe, generate_unique_filename


def test_add_suffix_safe_basic():
    p = Path('/tmp/abc')
    assert add_suffix_safe(p, '.dm5') == Path('/tmp/abc.dm5')
    assert add_suffix_safe(p, 'dm5') == Path('/tmp/abc.dm5')


def test_add_suffix_safe_dotted_name():
    # 含多个点的文件名：直接追加后缀
    p = Path('/tmp/a.b.c')
    assert add_suffix_safe(p, '.dm5') == Path('/tmp/a.b.c.dm5')


def test_add_suffix_safe_existing():
    p = Path('/tmp/abc.dm5')
    assert add_suffix_safe(p, '.dm5') == p
    assert add_suffix_safe(p, '.DM5') == p


def test_generate_unique_filename(tmp_path):
    # 实现从 (2) 开始编号
    target = Path(tmp_path) / 'test.dm5'
    target.touch()
    out = generate_unique_filename(target)
    assert Path(out).name == 'test(2).dm5'
    Path(out).touch()
    out2 = generate_unique_filename(target)
    assert Path(out2).name == 'test(3).dm5'