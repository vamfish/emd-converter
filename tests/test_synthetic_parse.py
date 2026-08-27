"""合成 EMD 骨架：9 类特征解析 + 导出冒烟测试。"""
import numpy as np
import pytest

from velox_file_analyzer2 import VeloxFileAnalyzer, dm5_writer


def analyze(path):
    a = VeloxFileAnalyzer(str(path))
    return a


@pytest.fixture
def build_all(tmp_path):
    from tests import synthetic_emd as s
    return s, tmp_path


# 每类特征：应产生的属性、数据形状原件（frames=1 时解析器对单帧切片为 2D）
CASES = [
    ('camera', ['camera'], 1, {'tem_data': (16, 16)}),
    ('camera_series', ['camera'], 5, {'tem_data': (16, 16, 5)}),
    ('stem', ['stem'], 1, {'stem_data': {'HAADF': (16, 16)}}),
    ('stem_series', ['stem'], 5, {'stem_data': {'HAADF': (16, 16, 5)}}),
    # DPC/crop/filter/dcfi 解析器取 [:, :, :]，恒为 3D
    ('dpc', ['dpc'], 1, {'dpc_data': {'HAADF': (16, 16, 1),
                                      'DF': (16, 16, 1),
                                      'iDPC': (16, 16, 1)}}),
    ('dcfi', ['dcfi'], 1, {'dcfi_data': (16, 16, 1)}),
    ('crop', ['crop'], 1, {'crop_data': (16, 16, 1)}),
    ('filter', ['filter'], 1, {'filter_data': (16, 16, 1)}),
    ('integrated', ['integrated', 'camera'], 1, {'spectra_data': {}}),
    ('si', ['si'], 1, {'mapping_data': {'HAADF': (16, 16), 'O': (16, 16), 'Al': (16, 16)},
                       'spectra_data': {'total': (32,)}}),
    ('colormix', ['si', 'colormix'], 1, {'line_profile_data': {'HAADF': None}}),
]


@pytest.mark.parametrize('name,feats,frames,checks', CASES, ids=[c[0] for c in CASES])
def test_parse_and_export(tmp_path, name, feats, frames, checks):
    from tests import synthetic_emd as s
    f = s.build_emd(tmp_path / f'{name}.emd', features=feats, shape=(16, 16), frames=frames)
    a = analyze(f)

    # 解析结果形状检查
    for attr, expected in checks.items():
        assert hasattr(a, attr), f'{attr} 缺失'
        val = getattr(a, attr)
        if isinstance(val, dict):
            for k, shape in expected.items():
                assert k in val, f'{attr}[{k}] 缺失 (有 {list(val.keys())})'
                if shape is None or (hasattr(shape, '__len__') and shape == ()):
                    continue
                if isinstance(val[k], np.ndarray):
                    assert val[k].shape == shape, f'{attr}[{k}] shape {val[k].shape} != {shape}'
        elif isinstance(val, np.ndarray):
            assert val.shape == expected, f'{attr} shape {val.shape} != {expected}'

    # 关键参数存在（顶层或嵌套于各探测器字典）
    flat = {k: v for k, v in a.parameters.items() if not isinstance(v, dict)}
    nested = [v for v in a.parameters.values() if isinstance(v, dict)]
    merged = dict(flat)
    for d in nested:
        if isinstance(d, dict):
            merged.update({k: v for k, v in d.items() if k in ('pixelsize', 'pixelunit')})
    for key in ('pixelsize', 'pixelunit'):
        assert key in merged, f'parameters 缺少 {key} (有 {list(a.parameters.keys())})'

    # 通用低风险导出：DM5 导出（tem/stem/dpc）
    exported = []
    if hasattr(a, 'tem_data') and a.tem_data.ndim in (2, 3):
        exported.append(_export_generic(a, 'Ceta', a.tem_data, a.tem_metadata, tmp_path, 'tem'))
    if hasattr(a, 'stem_data'):
        for k, v in a.stem_data.items():
            exported.append(_export_generic(a, k, v, a.stem_metadata.get(k, {}), tmp_path, 'stem'))
    if hasattr(a, 'dpc_data'):
        for k, v in a.dpc_data.items():
            exported.append(_export_generic(a, k, v, a.dpc_metadata.get(k, {}), tmp_path, 'dpc'))
    for p in exported:
        assert p.exists() and p.stat().st_size > 0
    a.f.close()


def _export_generic(a, key, data, metadata, tmp_path, tag):
    from velox_file_analyzer2 import dm5_writer
    import velox_file_analyzer2
    from velox_file_analyzer2 import add_suffix_safe
    params = a.parameters.get(key, {})
    if not isinstance(params, dict):
        params = {}
    signal = {'data': data, 'metadata': metadata,
              'color': {'red': 1, 'green': 1, 'blue': 1},
              'display_range': [0, 1], 'gamma': 1.0}
    d = data
    if d.ndim == 2:
        d = d[..., np.newaxis]
    signal['data'] = d
    out = tmp_path / f'{tag}_{key}.dm5'
    dm5_writer(out, signal, params if params else a.parameters)
    return out


def test_si_colormix_line_profile_shape(build_all):
    """SI + ColorMix 组合：线剖面数据形状正确。"""
    s, tmp = build_all
    f = s.build_emd(tmp / 'si_cm.emd', features=['si', 'colormix'], shape=(64, 64))
    a = analyze(f)
    assert hasattr(a, 'line_position')
    n_samples = a.line_position['length']
    for key, prof in a.line_profile_data.items():
        assert prof['profile_2d'].shape[0] >= 10
        assert prof['profile_avg'].shape[0] == prof['profile_2d'].shape[0]
    assert hasattr(a, 'color_mix_image')
    assert a.color_mix_image.shape == (64, 64, 3)
    a.f.close()