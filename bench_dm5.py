#!/usr/bin/env python
"""DM5-only 转换基准测试：复刻 emd_converter_gui.py 中 DM5 导出路径。

用法: python bench_dm5.py [--json results.json]
输出: temp/dm5_bench/{源文件名}/ 下 1 个或多个 dm5 文件
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from velox_file_analyzer2 import VeloxFileAnalyzer, dm5_writer

TEST_DATA = Path("test_data")
OUT_BASE = Path("temp") / "dm5_bench"


def add_suffix_safe(output_path: Path, suffix: str) -> Path:
    """与 emd_converter_gui.add_suffix_safe 保持一致。"""
    if not suffix.startswith("."):
        suffix = "." + suffix
    name = output_path.name
    if name.lower().endswith(suffix.lower()):
        return output_path
    return output_path.parent / (name + suffix)


def export_generic_image(analyzer, output_dir, filename_stem, param_key, data,
                         metadata, suffix=None, custom_name=None):
    """复刻 GUI._export_generic_image，仅 DM5。"""
    params = analyzer.parameters.get(param_key, {})
    if not isinstance(params, dict):
        params = {}

    if custom_name:
        filename = custom_name
    elif suffix:
        filename = f"{filename_stem}-{suffix}"
    else:
        filename = filename_stem
    output_path = output_dir / filename

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
    dm5_writer(add_suffix_safe(output_path, ".dm5"), signal,
               params if params else analyzer.parameters)
    return output_path


def export_eds_mapping(analyzer, output_dir, filename_stem):
    """复刻 GUI._export_eds_mapping，仅 DM5。"""
    quantification_mode = analyzer.parameters.get("quantification_mode", "Unknown")
    exported = []
    for key, value in analyzer.mapping_data.items():
        is_haadf = "HAADF" in key or "BF" in key or "DF" in key
        if is_haadf or True:  # GUI 默认 eds_options: export_haadf=True, export_elements=True
            filename = f"{filename_stem}-{key}-{quantification_mode}"
            data = value["data"][:, :, value["frame_index"]]
            output_path = output_dir / filename
            dm5_writer(add_suffix_safe(output_path, ".dm5"), value, analyzer.parameters)
            exported.append(str(add_suffix_safe(output_path, ".dm5")))
    return exported


def export_by_type(analyzer, output_dir, filename_stem):
    """复刻 GUI._export_by_type，只有 DM5 选项开启。"""
    written = []
    if hasattr(analyzer, "si_feature_path"):
        written += export_eds_mapping(analyzer, output_dir, filename_stem)
    if hasattr(analyzer, "camera_feature_path"):
        p = export_generic_image(analyzer, output_dir, filename_stem,
                                 "Ceta", analyzer.tem_data, analyzer.tem_metadata)
        written.append(str(p))
    if hasattr(analyzer, "stem_feature_path"):
        for key in analyzer.stem_data.keys():
            p = export_generic_image(analyzer, output_dir, filename_stem,
                                     key, analyzer.stem_data[key],
                                     analyzer.stem_metadata.get(key, {}), suffix=key)
            written.append(str(p))
    if hasattr(analyzer, "dpc_feature_path"):
        for key in analyzer.dpc_data.keys():
            p = export_generic_image(analyzer, output_dir, filename_stem,
                                     key, analyzer.dpc_data[key],
                                     analyzer.dpc_metadata.get(key, {}), suffix=key)
            written.append(str(p))
    if hasattr(analyzer, "dcfi_feature_path"):
        params = analyzer.parameters.get("DCFI", {})
        image_name = params.get("image_name", f"{filename_stem}-DCFI")
        p = export_generic_image(analyzer, output_dir, filename_stem,
                                 "DCFI", analyzer.dcfi_data, {}, custom_name=image_name)
        written.append(str(p))
    if hasattr(analyzer, "crop_feature_path"):
        params = analyzer.parameters.get("crop", {})
        image_name = params.get("image_name", f"{filename_stem}-Crop")
        p = export_generic_image(analyzer, output_dir, filename_stem,
                                 "crop", analyzer.crop_data, analyzer.crop_metadata,
                                 custom_name=image_name)
        written.append(str(p))
    if hasattr(analyzer, "image_filter_feature_path"):
        from urllib.parse import unquote
        params = analyzer.parameters.get("filter", {})
        image_name = params.get("image_name", f"{filename_stem}-Filtered")
        p = export_generic_image(analyzer, output_dir, filename_stem,
                                 "filter", analyzer.filter_data, analyzer.filter_metadata,
                                 custom_name=image_name)
        written.append(str(p))
    return written


def process_one(file_path: Path, out_dir: Path) -> dict:
    """处理单个文件，返回耗时信息。"""
    t0 = time.perf_counter()
    analyzer = VeloxFileAnalyzer(str(file_path))
    t_parse = time.perf_counter() - t0

    stem = file_path.stem
    file_output_dir = out_dir / stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    t1 = time.perf_counter()
    written = export_by_type(analyzer, out_dir, stem)
    t_export = time.perf_counter() - t1
    analyzer.f.close()

    return {
        "file": file_path.name,
        "size_gb": round(file_path.stat().st_size / 2**30, 3),
        "parse_s": round(t_parse, 2),
        "export_s": round(t_export, 2),
        "total_s": round(t_parse + t_export, 2),
        "dm5_files": len(written),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="结果输出到 JSON 文件")
    parser.add_argument("--files", nargs="*", default=None,
                        help="指定部分文件（默认全部 14 个）")
    args = parser.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = sorted(TEST_DATA.glob("*.emd"))
    print(f"共 {len(files)} 个文件", flush=True)

    results = []
    t_start = time.perf_counter()
    out_json = Path(args.json) if args.json else None
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files, 1):
        size = f.stat().st_size / 2**30
        print(f"\n[{i}/{len(files)}] {f.name} ({size:.2f} GB)", flush=True)
        try:
            r = process_one(f, OUT_BASE)
            results.append(r)
            print(f"    解析 {r['parse_s']}s | 导出 {r['export_s']}s | "
                  f"合计 {r['total_s']}s | DM5 ×{r['dm5_files']}", flush=True)
        except Exception as e:
            import traceback
            print(f"    [错误] {e}\n{traceback.format_exc()}", flush=True)
            results.append({"file": f.name, "error": str(e)})
        if out_json:  # 增量保存，避免中断丢失
            out_json.write_text(
                json.dumps({"files": results}, ensure_ascii=False, indent=2),
                encoding="utf-8")

    total = time.perf_counter() - t_start
    ok = [r for r in results if "error" not in r]
    print("\n" + "=" * 60, flush=True)
    print(f"总计耗时: {total:.2f} s ({total/60:.2f} min)", flush=True)
    if ok:
        sum_parse = sum(r["parse_s"] for r in ok)
        sum_export = sum(r["export_s"] for r in ok)
        print(f"其中: 解析 {sum_parse:.2f}s | 导出 {sum_export:.2f}s", flush=True)
        print(f"成功 {len(ok)}/{len(results)} 个文件", flush=True)

    summary = {
        "total_s": round(total, 2),
        "total_min": round(total / 60, 2),
        "sum_parse_s": round(sum_parse, 2) if ok else 0,
        "sum_export_s": round(sum_export, 2) if ok else 0,
        "files": results,
    }
    if out_json:
        out_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已保存: {out_json}", flush=True)


if __name__ == "__main__":
    main()