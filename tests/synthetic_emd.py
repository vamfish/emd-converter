"""构造最小可解析的 Velox EMD 骨架文件。

用于 pytest（CI）：覆盖 9 类特征（camera/stem/dpc/dcfi/crop/filter/
si/integrated/colormix），验证解析与导出管道在最小但结构合法的输入上不崩溃、
输出正确。真实 14 文件的逐字节一致性回归仍由根目录 compare_dm5.py 负责。

结构依据 velox_file_analyzer2.py 中 VeloxFeature 各处理器的实际读取路径复刻。
"""
import json
import uuid
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np


def _uid() -> str:
    return uuid.uuid4().hex


# --------------------------------------------------------------------------
# 基础写入工具
# --------------------------------------------------------------------------

def put_json(f: h5py.File, path: str, obj) -> None:
    """写入 shape (1,), dtype object 的 JSON dataset（bytes_to_json 可读）。"""
    f.create_dataset(path, data=np.array([json.dumps(obj).encode('utf-8')], dtype=object))


def put_metadata(f: h5py.File, path: str, meta) -> None:
    """写入 decode_metadata 要求的 (60000, 1) uint8 元数据（JSON 文本 + 零填充）。"""
    payload = json.dumps(meta, ensure_ascii=True).encode('ascii')
    arr = np.zeros((60000, 1), dtype=np.uint8)
    b = np.frombuffer(payload, dtype=np.uint8)
    arr[: len(b), 0] = b
    f.create_dataset(path, data=arr)


def make_metadata(pixel_size: str = '1.0e-9', pixel_unit: str = 'm',
                  detector: str = 'Det-1', frames: int = 1) -> dict:
    """构造解析/导出所需的最小元数据字典。"""
    return {
        'Core': {}, 'Instrument': {}, 'Acquisition': {}, 'Optics': {},
        'EnergyFilter': {}, 'Stage': {}, 'Scan': {'DwellTime': '0.0001'},
        'Vacuum': {}, 'Sample': {},
        'Detectors': {
            detector: {
                'RealTime': '100.0', 'LiveTime': '50.0',
                'InputCountRate': '1000', 'OutputCountRate': '1000',
                'OffsetEnergy': '-250', 'BeginEnergy': '-250',
                'EndEnergy': '20480', 'DetectorName': detector,
            }
        },
        'BinaryResult': {
            'AcquisitionUnit': 'CameraImage', 'CompositionType': '',
            'ImageSize': {'width': '16', 'height': '16'},
            'DetectorIndex': '0', 'Detector': detector,
            'PixelSize': {'width': pixel_size, 'height': pixel_size},
            'PixelUnitX': pixel_unit, 'PixelUnitY': pixel_unit,
            'Offset': {'x': '0.0', 'y': '0.0'}, 'DwellTime': '10',
        },
        'CustomProperties': {
            'Velox.Series.FrameNumber': {'value': str(frames)},
            'Scan': {'DwellTime': '0.0001'},
        },
    }


def make_image(f: h5py.File, data_path: str, arr: np.ndarray,
               meta: Optional[dict] = None) -> None:
    """写入 /Data/Image/<uid> 下的 Data + FrameLookupTable + Metadata。"""
    f.create_dataset(data_path + '/Data', data=arr)
    frames = arr.shape[2] if arr.ndim == 3 else 1
    f.create_dataset(data_path + '/FrameLookupTable',
                     data=np.arange(frames, dtype=np.uint32))
    put_metadata(f, data_path + '/Metadata',
                 meta or make_metadata(detector='Det-1', frames=frames))


def make_display(f: h5py.File, data_path: str, label: str = 'Img',
                 series_index: int = 0, gamma: str = '1',
                 begin: str = '0', end: str = '1000',
                 disp_id: str = 'a', extra: Optional[dict] = None) -> str:
    """写入 ImageDisplay object JSON，返回其路径。"""
    path = f'/Presentation/Displays/ImageDisplay/{_uid()}'
    obj = {
        'display': {'id': disp_id, 'label': label, 'overlays': '',
                    'priority': '1', 'closable': 'false', 'visible': 'true'},
        'dataPath': data_path,
        'seriesIndex': str(series_index),
        'zoom': '1', 'pan': {'x': '0', 'y': '0'}, 'angle': '0', 'offsetAngle': '0',
        'cropMode': 'PowerOf2', 'autoDisplayLevels': 'true', 'autoWhiteMode': 'false',
        'displayLevelsRange': {'begin': begin, 'end': end},
        'invertColors': 'false', 'gamma': gamma, 'vibrant': 'false',
        'colorLut': 'Grayscale', 'scaleBarVisibility': 'Visible',
        'scaleBarAlignment': 'BottomLeft', 'dataBarVisibility': 'Hidden',
        'interpolationMode': 'Auto',
    }
    if extra:
        obj.update(extra)
    put_json(f, path, obj)
    return path


def default_image(shape: Tuple[int, int], frames: int = 1, dtype=None) -> np.ndarray:
    """生成确定性的测试图像数据（值 = 像素序号），恒为 3D (H, W, F)。
    
    注意：解析器对单帧/多帧的 Data 一律做 3 维索引（[:, :, 0] 或 [:, :, :]），
    因此骨架中的 Data 必须恒为 3D。
    """
    h, w = shape
    arr = (np.arange(h * w * frames, dtype=np.int16) % 1000).reshape(h, w, frames)
    return arr.astype(dtype) if dtype else arr


# --------------------------------------------------------------------------
# 特征构造器（每个返回注册到 Features 的 dict）
# --------------------------------------------------------------------------

def add_camera(f: h5py.File, shape: Tuple[int, int], frames: int = 1) -> dict:
    """TEM / CameraFeature（单张与系列共用，frames 区分）。"""
    data_path = f'/Data/Image/{_uid()}'
    make_image(f, data_path, default_image(shape, frames), make_metadata(detector='Ceta', frames=frames))
    feat_path = f'/Features/CameraFeature/{_uid()}'
    op_path = f'/Operations/CameraInputOperation/{_uid()}'
    disp_path = make_display(f, data_path, label='Ceta')
    put_json(f, feat_path, {
        'cameraInputOperation': op_path,
        'imageDisplay': disp_path,
        'remoteStoragePath': '',
    })
    put_json(f, op_path, {'dataPath': data_path, 'cameraName': 'Ceta', 'outputs': ''})
    return {'CameraFeature': feat_path}


def add_stem(f: h5py.File, shape: Tuple[int, int], frames: int = 1) -> dict:
    """STEMFeature（单张与系列共用）。"""
    data_path = f'/Data/Image/{_uid()}'
    make_image(f, data_path, default_image(shape, frames), make_metadata(detector='HAADF', frames=frames))
    feat_path = f'/Features/STEMFeature/{_uid()}'
    op_path = f'/Operations/StemInputOperation/{_uid()}'
    dlo_path = f'/Operations/DisplayLevelsOperation/{_uid()}'
    put_json(f, dlo_path, {})
    disp_path = make_display(f, data_path, label='HAADF')
    put_json(f, feat_path, {
        'stemInputOperations': [op_path],
        'displayLevelsOperations': [dlo_path],
        'imageDisplays': [disp_path],
    })
    put_json(f, op_path, {
        'dataPath': data_path,
        'detector': 'HAADF',
        'detectorInfo': {'name': 'HAADF', 'segments': ''},
        'outputs': [{'outputIndex': '0', 'operation': dlo_path, 'inputIndex': '0'}],
        'scanArea': ['0', '0', '1', '1'],
    })
    return {'STEMFeature': feat_path}


def add_dpc(f: h5py.File, shape: Tuple[int, int], segments: Sequence[str],
            frames: int = 1) -> dict:
    """DPCFeature：每个 segment 一张图（imageDisplays 与 image 一一对应）。"""
    feat_path = f'/Features/DPCFeature/{_uid()}'
    ops, dlos, disps = [], [], []
    for i, label in enumerate(segments):
        data_path = f'/Data/Image/{_uid()}'
        make_image(f, data_path, default_image(shape, frames),
                   make_metadata(detector=label, frames=frames))
        op_path = f'/Operations/StemInputOperation/{_uid()}'
        dlo_path = f'/Operations/DisplayLevelsOperation/{_uid()}'
        put_json(f, dlo_path, {})
        disp_path = make_display(f, data_path, label=label, series_index=0)
        put_json(f, op_path, {'dataPath': data_path, 'detector': label})
        ops.append(op_path)
        dlos.append(dlo_path)
        disps.append(disp_path)
    put_json(f, feat_path, {
        'stemInputOperations': ops,
        'displayLevelsOperations': dlos,
        'imageDisplays': disps,
    })
    return {'DPCFeature': feat_path}


def add_dcfi(f: h5py.File, shape: Tuple[int, int], frames: int = 1) -> dict:
    """DcfiFeature：imageDisplay 指向经过漂移校正的数据。"""
    data_path = f'/Data/Image/{_uid()}'
    make_image(f, data_path, default_image(shape, frames), make_metadata(detector='Ceta', frames=frames))
    feat_path = f'/Features/DcfiFeature/{_uid()}'
    shp = f'/Operations/ShiftMeasurementOperation/{_uid()}'
    integ = f'/Operations/IntegrationOperation/{_uid()}'
    dlo = f'/Operations/DisplayLevelsOperation/{_uid()}'
    disp_path = make_display(f, data_path, label='DCFI(Ceta)')
    put_json(f, shp, {})
    put_json(f, integ, {})
    put_json(f, dlo, {})
    put_json(f, feat_path, {
        'shiftMeasurementOperation': shp,
        'integrationOperation': integ,
        'displayLevelsOperation': dlo,
        'imageDisplay': disp_path,
    })
    return {'DcfiFeature': feat_path}


def add_crop(f: h5py.File, shape: Tuple[int, int], frames: int = 1) -> dict:
    """CropFeature：裁剪图像 + 标注形状。"""
    data_path = f'/Data/Image/{_uid()}'
    make_image(f, data_path, default_image(shape, frames), make_metadata(detector='Ceta', frames=frames))
    feat_path = f'/Features/CropFeature/{_uid()}'
    crop_op = f'/Operations/CropOperation/{_uid()}'
    shape_path = f'/SharedProperties/AnnotationShape/{_uid()}'
    # 标注: {dataPath: 形状数据, color: 颜色}  —— 与真实 Annotation 结构一致
    ann = f'/Presentation/Overlays/Annotation/{_uid()}'
    put_json(f, shape_path, {'shape': 'rectangle', 'x': 0, 'y': 0, 'w': 8, 'h': 8})
    put_json(f, ann, {'dataPath': shape_path,
                      'color': {'red': '1', 'green': '0', 'blue': '0'}})
    disp_path = make_display(f, data_path, label='Crop#1')
    put_json(f, feat_path, {
        'imageDisplay': disp_path,
        'cropOperationPath': crop_op,
        'cropAnnotationPath': ann,
        'inputSize': {'width': '0', 'height': '0'},
    })
    put_json(f, crop_op, {})
    return {'CropFeature': feat_path}


def add_filter(f: h5py.File, shape: Tuple[int, int], frames: int = 1) -> dict:
    """ImageFilteringFeature：滤波图像 + 滤波设置。"""
    data_path = f'/Data/Image/{_uid()}'
    make_image(f, data_path, default_image(shape, frames), make_metadata(detector='Ceta', frames=frames))
    feat_path = f'/Features/ImageFilteringFeature/{_uid()}'
    record = f'/Operations/ImageFilteringOperation/{_uid()}'
    settings = f'/SharedProperties/ImageFilteringSettings/{_uid()}'
    put_json(f, settings, {'mode': 'median', 'size': '3'})
    disp_path = make_display(f, data_path, label='Filtered#1')
    put_json(f, record, {'settingsPath': settings, 'filterType': 'MedianFilter'})
    put_json(f, feat_path, {
        'imageDisplay': disp_path,
        'imageFilteringOperationRecord': record,
    })
    return {'ImageFilteringFeature': feat_path}


def add_integrated_spectra(f: h5py.File) -> dict:
    """IntegratedSpectraFeature：积分能谱（依赖 Data/Spectrum 可选）。"""
    feat_path = f'/Features/IntegratedSpectraFeature/{_uid()}'
    ann = f'/SharedProperties/Annotation/{_uid()}'
    disp = f'/Displays/SpectrumDisplay/{_uid()}'
    in_disp = f'/Displays/ImageDisplay/{_uid()}'
    qs = f'/SharedProperties/EDSSpectrumQuantificationSettings/{_uid()}'
    bg = f'/SharedProperties/BackgroundCorrection/{_uid()}'
    bw = f'/SharedProperties/BackgroundWindows/{_uid()}'
    put_json(f, bg, {'model': 'None'})
    put_json(f, bw, {'backgroundWindows': [{'begin': 1, 'end': 10}]})
    put_json(f, qs, {
        'backgroundCorrection': bg,
        'absorptionCorrection': {'density': 1, 'enabled': False,
                                 'sampleThickness': 5e-08, 'useDensity': False},
        'backgroundWindows': bw,
        'elementProperties': [{'element': 8, 'quantify': True},
                              {'element': 13, 'quantify': True}],
        'elementSelection': [0, 1],
        'ionizationCrossSectionModel': 'K',
    })
    put_json(f, ann, {})
    put_json(f, disp, {})
    put_json(f, in_disp, {})
    put_json(f, feat_path, {
        'annotations': [ann],
        'display': disp,
        'inputDisplay': in_disp,
        'quantificationSettings': qs,
    })
    return {'IntegratedSpectraFeature': feat_path}


def add_spectrum(f: h5py.File, channels: int = 32) -> str:
    """写入 Data/Spectrum/<uid> 谱图数据（integrated spectra 用）。"""
    gid = _uid()
    data_path = f'/Data/Spectrum/{gid}'
    f.create_dataset(data_path + '/Data', data=np.ones((channels, 1), dtype=np.float32))
    put_metadata(f, data_path + '/Metadata', make_metadata(detector='EDS1'))
    return data_path


def add_si(f: h5py.File, shape: Tuple[int, int], elements: Sequence[str] = ('O', 'Al'),
           channels: int = 32) -> dict:
    """SIFeature（EDS Mapping）：mapping 显示项 + 谱图 + 定量设置。"""
    h, w = shape
    feat_path = f'/Features/SIFeature/{_uid()}'

    # --- mapping displayGroupItems：每个元素 / HAADF 一个 ---
    # 结构对齐真实文件：
    #   display = {'data': 路径, 'id': 标签, 'settings': 路径, 'title': 标签}
    #   dgi     = {'blendFactor': 数字, 'blendMode': 'Alpha', 'display': 路径,
    #              'groupType': 'Stem'|..., 'name': 标签, 'selected': True}
    mid = f'/SharedProperties/MultiImageDisplay/{_uid()}'
    items: List[str] = []
    # HAADF（Stem groupType，提供 image_shape/frames）
    for i, (label, group_type) in enumerate([('HAADF', 'Stem')] +
                                            [(el, 'EDS') for el in elements]):
        data_path = f'/Data/Image/{_uid()}'
        make_image(f, data_path, default_image(shape), make_metadata(detector=label, frames=1))
        data_obj = f'/SharedProperties/ImageSeriesDataReference/{_uid()}'
        put_json(f, data_obj, {'dataPath': data_path, 'frameIndex': '0'})
        settings = f'/SharedProperties/ImageDisplaySettings/{_uid()}'
        put_json(f, settings, {'color': {'red': 1, 'green': 1, 'blue': 1},
                               'displayLevelsRange': {'begin': '0', 'end': '1000'},
                               'gamma': '1'})
        disp = f'/Displays/ImageDisplay/{_uid()}'
        put_json(f, disp, {'data': data_obj, 'id': label,
                           'settings': settings, 'title': label})
        dgi = f'/SharedProperties/DisplayGroupItem/{_uid()}'
        put_json(f, dgi, {'blendFactor': 1, 'blendMode': 'Alpha',
                          'display': disp, 'groupType': group_type,
                          'name': label, 'selected': True})
        items.append(dgi)
    put_json(f, mid, {'displayGroupItems': items, 'id': 'mid', 'title': 'mapping'})

    # --- EDS 谱图（detectors -> segments -> renderedSpectrum）---
    rend = f'/SharedProperties/DataReference/{_uid()}'
    spec_img_path = f'/Data/SpectrumImage/{_uid()}'
    put_json(f, rend, {'dataPath': spec_img_path})
    f.create_dataset(spec_img_path + '/SpectrumImageSettings',
                     data=np.array([json.dumps({'channels': channels}).encode()], dtype=object))
    f.create_dataset(spec_img_path + '/Data',
                     data=np.ones((1,), dtype=np.float32))
    put_metadata(f, spec_img_path + '/Metadata', make_metadata(detector='EDS1'))
    eds_seg = f'/SharedProperties/DataReference/{_uid()}'
    spec_data = f'/Data/Spectrum/{_uid()}'
    f.create_dataset(spec_data + '/Data', data=np.ones((channels, 1), dtype=np.float32))
    put_metadata(f, spec_data + '/Metadata',
                 make_metadata(detector='EDS1', frames=1))
    put_json(f, eds_seg, {'dataPath': spec_data})

    qs = f'/SharedProperties/EDSQuantificationSettings/{_uid()}'
    bg = f'/SharedProperties/ModeledBackgroundCorrectionModel/{_uid()}'
    filt = f'/SharedProperties/EDSSpectrumFilterSettings/{_uid()}'
    put_json(f, bg, {'model': 'None'})
    put_json(f, filt, {'filter': 'None'})
    put_json(f, qs, {
        'backgroundCorrection': bg,
        'absorptionCorrection': {'density': 1, 'enabled': False,
                                 'sampleThickness': 5e-08, 'useDensity': False},
        'elementProperties': [{'element': 8, 'quantify': True, 'atomicShellFamily': 'K'},
                              {'element': 13, 'quantify': True, 'atomicShellFamily': 'K'}],
        'elementSelection': [0, 1],
        'ionizationCrossSectionModel': 'K',
    })

    eds = {
        'detectors': [{'physicalDetector': 'EDS1',
                       'segments': [{'index': '0', 'summed': True,
                                     'renderedSpectrum': eds_seg,
                                     'spectrumStream': ''}]}],
        'quantificationKernelFilterSettings': 'None',
        'quantificationSettings': qs,
        'spectralFiltersettings': filt,
        'spectrumImage': rend,
    }
    color_mix = f'/Presentation/Displays/ImageDisplay/{_uid()}'
    put_json(f, color_mix, {'label': 'ColorMix'})
    put_json(f, feat_path, {
        'colorMixDisplay': color_mix,
        'eds': eds,
        'multiImageDisplay': mid,
        'quantificationMode': 'NetIntensity',
        'stem': {'detector': 'HAADF'},
    })
    return {'SIFeature': feat_path}


def add_colormix(f: h5py.File, shape: Tuple[int, int]) -> dict:
    """ColorMixProfileFeature：线剖面 + 颜色混合。
    
    依赖 SI 特征先构造 mapping_data（feature_handlers 顺序保证）；
    这里只需提供线标注（annotation）与输入数据引用。
    """
    feat_path = f'/Features/ColorMixProfileFeature/{_uid()}'
    ann = f'/Presentation/Overlays/Annotation/{_uid()}'
    shape_json = f'/SharedProperties/AnnotationShape/{_uid()}'
    appearance = f'/SharedProperties/AnnotationAppearance/{_uid()}'
    input_data = f'/SharedProperties/DataReference/{_uid()}'
    h, w = shape
    # 对角线上的线，全部采样点都落在图像内
    put_json(f, shape_json, {'line': {
        'p1': {'x': 0.1, 'y': 0.1},
        'p2': {'x': 0.8, 'y': 0.8},
    }})
    put_json(f, appearance, {'lineSettings': {'width': '3'}})
    put_json(f, ann, {'shape': shape_json, 'appearance': appearance})
    put_json(f, input_data, {'displayGroupItems': []})
    put_json(f, feat_path, {
        'annotation': ann,
        'imageInputData': input_data,
    })
    return {'ColorMixProfileFeature': feat_path}


def build_emd(path, features=('camera',), shape=(16, 16), frames=1):
    """构造最小 EMD 文件。

    features: 'camera'|'stem'|'dpc'|'dcfi'|'crop'|'filter'|'si'|'integrated'
              （'colormix' 需与 'si' 组合）
    """
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    charters = {
        'camera': lambda f: add_camera(f, shape, frames),
        'stem': lambda f: add_stem(f, shape, frames),
        'dpc': lambda f: add_dpc(f, shape, ('HAADF', 'DF', 'iDPC')),
        'dcfi': lambda f: add_dcfi(f, shape, frames),
        'crop': lambda f: add_crop(f, shape, frames),
        'filter': lambda f: add_filter(f, shape, frames),
        'integrated': lambda f: add_integrated_spectra(f),
        'si': lambda f: add_si(f, shape),
        'colormix': lambda f: add_colormix(f, shape),
    }
    with h5py.File(path, 'w') as f:
        feature_list = [charters[name](f) for name in features]
        put_json(f, 'Features/Features', {'features': feature_list})
        # Experiment 日志
        log_id = _uid()
        put_json(f, 'Experiment', {'log': f'/Data/Text/{log_id}'})
        put_json(f, f'/Data/Text/{log_id}', {'text': '<p>synthetic log</p>'})
        # 常见顶层结构（部分代码会反射检查）
        f.create_dataset('Info', data=np.array([json.dumps({}).encode()], dtype=object))
        put_json(f, 'Presentation/DisplayIndex', {})
        put_json(f, 'Operations/Operations', {})
        put_json(f, 'Version', {'major': 1})
    return path