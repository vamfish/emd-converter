"""拖放路径解析与 EMD 收集测试（纯函数，无需显示环境）。"""
from pathlib import Path

from emd_converter_gui import parse_drop_paths, collect_emd_files


def test_parse_braced_windows_paths():
    raw = '{D:/TEM Data/a b/x.emd} {D:/y.emd}'
    assert parse_drop_paths(raw) == ['D:/TEM Data/a b/x.emd', 'D:/y.emd']


def test_parse_plain_space_separated():
    assert parse_drop_paths('/tmp/a.emd /tmp/b.emd') == ['/tmp/a.emd', '/tmp/b.emd']


def test_parse_file_uris():
    assert parse_drop_paths('file:///tmp/a%20b.emd') == ['/tmp/a b.emd']
    # Windows 形态 URI 去掉开头斜杠
    assert parse_drop_paths('file:///D:/x/a.emd') == ['D:/x/a.emd']


def test_parse_mixed_and_empty():
    assert parse_drop_paths('') == []
    assert parse_drop_paths('{C:/a b.emd} /tmp/c.emd') == ['C:/a b.emd', '/tmp/c.emd']


def test_collect_expands_dirs_dedups_and_case_insensitive(tmp_path):
    (tmp_path / 'one.emd').write_bytes(b'x')
    (tmp_path / 'note.txt').write_text('no')
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'three.EMD').write_bytes(b'x')  # 大写后缀

    got = collect_emd_files([
        str(tmp_path),                    # 目录递归：one + three
        str(tmp_path / 'one.emd'),        # 与目录展开重复 → 去重
        str(tmp_path / 'note.txt'),       # 非 emd → 忽略
    ])
    names = sorted(Path(p).name.lower() for p in got)
    assert names == ['one.emd', 'three.emd']


def test_collect_ignores_missing_paths(tmp_path):
    assert collect_emd_files([str(tmp_path / 'ghost.emd'), '']) == []
