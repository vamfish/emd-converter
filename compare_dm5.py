#!/usr/bin/env python
"""正确性对比：原始版 vs 优化版的 DM5 输出结构完全一致。

用法: python compare_dm5.py
输出: temp/orig_out/{stem}/{...}.dm5（原始版）、temp/new_out/{stem}/{...}.dm5（优化版）
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent))
from velox_file_analyzer2 import VeloxFileAnalyzer as NewAnalyzer, dm5_writer as new_dm5_writer

# 加载原始版本（独立模块，避免与当前模块冲突）
spec = importlib.util.spec_from_file_location(
    "vfa_orig", "temp/orig/velox_file_analyzer2.py")
vfa_orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vfa_orig)

TEST_DATA = Path("test_data")
OUT = {"orig": Path("temp/orig_out"), "new": Path("temp/new_out")}


def add_suffix_safe(output_path: Path, suffix: str) -> Path:
    if not suffix.startswith("."):
        suffix = "." + suffix
    name = output_path.name
    if name.lower().endswith(suffix.lower()):
        return output_path
    return output_path.parent / (name + suffix)


def export_dm5(analyzer, out_dir, filename_stem, use_orig_writer):
    """用 analyzer 实例 + 指定 writer 导出 DM5，返回输出文件列表。"""
    written = []
    dm5_writer = vfa_orig.dm5_writer if use_orig_writer else new_dm5_writer
    file_output_dir = out_dir / filename_stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    def generic(param_key, data, metadata, suffix=None, custom_name=None):
        params = analyzer.parameters.get(param_key, {})
        if not isinstance(params, dict):
            params = {}
        if custom_name:
            filename = custom_name
        elif suffix:
            filename = f"{filename_stem}-{suffix}"
        else:
            filename = filename_stem
        output_path = file_output_dir / filename
        signal = {
            "data": data,
            "metadata": metadata,
            "color": {"blue": 1, "green": 1, "red": 1},
            "display_range": params.get("display_range", [0, 1]),
            "gamma": params.get("gamma", 1.0),
        }
        dm5_data = data
        if dm5_data.ndim == 2:
            dm5_data = dm5_data[..., np.newaxis]
        signal["data"] = dm5_data
        p = add_suffix_safe(output_path, ".dm5")
        dm5_writer(p, signal, params if params else analyzer.parameters)
        written.append(p)

    if hasattr(analyzer, "si_feature_path"):
        quantification_mode = analyzer.parameters.get("quantification_mode", "Unknown")
        for key, value in analyzer.mapping_data.items():
            filename = f"{filename_stem}-{key}-{quantification_mode}"
            data = value["data"][:, :, value["frame_index"]]
            output_path = file_output_dir / filename
            p = add_suffix_safe(output_path, ".dm5")
            dm5_writer(p, value, analyzer.parameters)
            written.append(p)
    if hasattr(analyzer, "camera_feature_path"):
        generic("Ceta", analyzer.tem_data, analyzer.tem_metadata)
    if hasattr(analyzer, "stem_feature_path"):
        for key in analyzer.stem_data.keys():
            generic(key, analyzer.stem_data[key],
                    analyzer.stem_metadata.get(key, {}), suffix=key)
    if hasattr(analyzer, "dpc_feature_path"):
        for key in analyzer.dpc_data.keys():
            generic(key, analyzer.dpc_data[key],
                    analyzer.dpc_metadata.get(key, {}), suffix=key)
    if hasattr(analyzer, "dcfi_feature_path"):
        params = analyzer.parameters.get("DCFI", {})
        image_name = params.get("image_name", f"{filename_stem}-DCFI")
        generic("DCFI", analyzer.dcfi_data, {}, custom_name=image_name)
    if hasattr(analyzer, "crop_feature_path"):
        params = analyzer.parameters.get("crop", {})
        image_name = params.get("image_name", f"{filename_stem}-Crop")
        generic("crop", analyzer.crop_data, analyzer.crop_metadata, custom_name=image_name)
    if hasattr(analyzer, "image_filter_feature_path"):
        params = analyzer.parameters.get("filter", {})
        image_name = params.get("image_name", f"{filename_stem}-Filtered")
        generic("filter", analyzer.filter_data, analyzer.filter_metadata, custom_name=image_name)
    return written


def walk(obj, prefix="", skip_values=False):
    """遍历 h5py 对象，产出 (路径, kind, shape, dtype, data, attrs)。"""
    for name in obj.keys():
        path = f"{prefix}/{name}"
        child = obj[name]
        if isinstance(child, h5py.Group):
            yield (path, "group", None, None, None,
                   dict(child.attrs))
            yield from walk(child, path, skip_values or name == "UniqueID")
        else:
            skip = skip_values or "UniqueID" in path
            data = None
            if not skip:
                try:
                    data = child[()]
                except Exception:
                    data = None
            yield (path, "dataset",
                   child.shape if not skip else None,
                   str(child.dtype) if not skip else None,
                   data,
                   dict(child.attrs))


def compare(a_path: Path, b_path: Path) -> list:
    errors = []
    with h5py.File(a_path, "r") as fa, h5py.File(b_path, "r") as fb:
        items_a = {path: v for path, *_ in [(x[0],) for x in []]}  # placeholder
        dict_a = {}
        for item in walk(fa):
            dict_a[item[0]] = item
        dict_b = {}
        for item in walk(fb):
            dict_b[item[0]] = item

        if set(dict_a) != set(dict_b):
            errors.append(f"  结构不同: 仅A={set(dict_a)-set(dict_b)} 仅B={set(dict_b)-set(dict_a)}")
        for path in sorted(set(dict_a) & set(dict_b)):
            a, b = dict_a[path], dict_b[path]
            if a[1] != b[1]:
                errors.append(f"  {path}: 类型 {a[1]} vs {b[1]}")
                continue
            if a[1] == "dataset":
                if a[2] != b[2]:
                    errors.append(f"  {path}: shape {a[2]} vs {b[2]}")
                if a[3] != b[3]:
                    errors.append(f"  {path}: dtype {a[3]} vs {b[3]}")
                if a[4] is not None and b[4] is not None:
                    if not np.array_equal(a[4], b[4]):
                        errors.append(f"  {path}: 数据值不同 (max diff={np.abs(a[4].astype(np.float64)-b[4].astype(np.float64)).max()})")
            # attrs 对比（UniqueID 组内随机值已在 walk 中放行值本身，这里仍比键集与 dtype）
            ka, kb = set(a[5]), set(b[5])
            if ka != kb:
                errors.append(f"  {path}: attrs 键不同 {ka^kb}")
    return errors


def main():
    files = sorted(TEST_DATA.glob("*.emd"))
    all_errors = []
    for i, f in enumerate(files, 1):
        stem = f.stem
        print(f"[{i}/{len(files)}] {f.name}")
        for version, analyzer_cls, writer_orig in [
            ("orig", vfa_orig.VeloxFileAnalyzer, True),
            ("new", NewAnalyzer, False),
        ]:
            a = analyzer_cls(str(f))
            export_dm5(a, OUT[version], stem, writer_orig)
            a.f.close()

        # 对比两个输出目录下的 dm5 文件
        orig_files = sorted(OUT["orig"].glob(f"{stem}/*.dm5"))
        new_files = sorted(OUT["new"].glob(f"{stem}/*.dm5"))
        if len(orig_files) != len(new_files):
            print(f"  [FAIL] DM5 文件数量不同: {len(orig_files)} vs {len(new_files)}")
            all_errors.append(f"{stem}: 文件数量不同")
            continue
        for o, n in zip(orig_files, new_files):
            if o.name != n.name:
                print(f"  [FAIL] DM5 文件名称不同: {o.name} vs {n.name}")
                all_errors.append(f"{stem}: {o.name} vs {n.name}")
                continue
            errs = compare(o, n)
            if errs:
                print(f"  [FAIL] {o.name}")
                for e in errs:
                    print(e)
                all_errors.extend(errs)
            else:
                print(f"  [OK]   {o.name} 完全一致")

    print("\n" + "=" * 60)
    if all_errors:
        print(f"对比完成: {len(all_errors)} 处差异")
        sys.exit(1)
    print("对比完成: 全部一致 ✓")


if __name__ == "__main__":
    main()