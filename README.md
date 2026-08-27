# EMD Converter

FEI Velox EMD 文件批量转换工具，支持导出为 DM5、TIFF、PNG 和 CSV 格式。

## 功能特性

- **批量处理**: 自动处理文件夹中的所有 EMD 文件
- **多格式导出**: 支持 DM5、TIFF (16-bit)、PNG、CSV 格式
- **数据类型支持**: STEM、TEM、EDS Mapping / Line Scan、DPC、DCFI、SAD 衍射图 等
- **元数据保留**: 保留像素尺寸、单位、显示范围、Gamma 等校准信息
- **高性能**:
  - 线剖面提取向量化（EDS Line Scan 平均提速 ~70×）
  - EMD 大数据读取/写入采用缓存分块策略（5GB 级原位序列提速 ~3×），
    低内存机器自动切换流式模式（峰值内存 ~2× → ~1.1× 数据量）
  - 可选**并行处理**（GUI 复选框）：多核加速，worker 数按可用内存自适应
  - **EMD → DM5 批量转换总耗时提升约 7.6×**
    （14 类测试文件：初始 150.7s → 串行 26.6s → 并行 19.7s）
- **用户友好**: 图形界面，支持配置自动保存；并行处理默认关闭
- **跨平台**: Windows / Linux / macOS

## 安装

### 方法 1: 使用 uv（推荐）

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

### 方法 2: 使用 pip

```bash
pip install -r requirements.txt
```

### 方法 3: 使用 Conda

```bash
conda create -n emd_converter python=3.10
conda activate emd_converter
pip install -r requirements.txt
```

### 方法 4: Windows 免安装版本

从 [Releases](https://github.com/vamfish/emd-converter/releases) 下载 `EMD_Converter.exe`（单文件便携版，无需安装 Python），双击即可运行。
该版本由 GitHub Actions 在 Windows runner 上自动构建；推送 `v*` 标签时：

1. 运行 pytest 测试门禁（先测试，后构建）
2. 构建单文件 Windows exe
3. 自动提取 `CHANGELOG.md` 对应版本块并创建 Release

手动构建（不发布）：

```bash
gh workflow run "Build Windows EXE"
```

## 使用方法

### 1. 启动 GUI

```bash
python launch_gui.py
```

或:

```bash
python emd_converter_gui.py
```

### 2. 使用步骤

1. 点击"选择文件夹"（或"添加文件"）按钮，选择包含 EMD 文件的文件夹
2. 勾选需要导出的格式 (DM5、TIFF、PNG、CSV)；EDS 数据可展开 EDS 选项
3. 点击"开始处理"
4. 转换后的文件将保存在输出目录的子文件夹中（默认 `custom_export/{源文件名}/`）

## 文件说明

| 文件 | 说明 |
|------|------|
| `emd_converter_gui.py` | 主程序，图形界面 |
| `velox_file_analyzer2.py` | EMD 文件解析核心库 |
| `launch_gui.py` | 启动脚本 |
| `requirements.txt` | Python 依赖列表 |
| `bench_dm5.py` | 性能基准脚本（`python bench_dm5.py --json results.json`） |
| `compare_dm5.py` | DM5 输出一致性回归校验（原版 vs 新版逐字节对比） |
| `CHANGELOG.md` | 版本变更记录（版本块与 git tag 对齐，供自动发版提取） |
| `tests/` | pytest 测试套件（合成 EMD 骨架，覆盖 9 类特征） |
| `.github/workflows/build-windows-exe.yml` | 测试门禁 + Windows 单文件 exe 构建 + 自动发布流水线 |

## 支持的 EMD 数据类型

- **STEM / TEM 图像**（单张与系列）: 导出为 DM5、TIFF、PNG
- **EDS Mapping**（元素分布图）: 导出为 DM5、TIFF、PNG
- **EDS 能谱**（积分谱图）: 导出为 CSV、PNG 谱图
- **EDS Line Scan**: ColorMix 图像、Line Profile PNG 与 CSV
- **DPC（差分相位衬度）**: 导出为 DM5、TIFF、PNG
- **DCFI（漂移校正帧积分）**: 导出为 DM5、TIFF、PNG
- **SAD 衍射图**: 导出为 DM5、TIFF、PNG（比例尺正确标注 1/nm 倒数空间单位）

## 输出文件命名规则

转换后的文件保存在输出目录的子文件夹中，命名格式：
- `{原文件名}_{数据类型}_{编号}.{格式}`

例如：`sample_DF_0001.dm5`

## 注意事项

- TIFF 导出使用 16-bit 格式以保留完整动态范围
- 像素尺寸、单位等校准信息会保留在 TIFF 和 DM5 文件的元数据中
- 大文件（数 GB 级原位序列）处理需要较多内存；低内存机器会自动切换流式模式并给出提示
- 勾选"并行处理"可加速多文件批次（默认关闭；大文件按内存预算自动独占串行）
- 配置文件 `gui_config.json` 会自动保存用户设置

## 系统要求

- Python 3.8+
- Windows / Linux / macOS

## 依赖列表

核心依赖：
- numpy >= 1.20.0
- h5py >= 3.0.0
- tifffile >= 2021.0.0
- Pillow >= 8.0.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- beautifulsoup4 >= 4.9.0
- tqdm >= 4.60.0

## 许可证

MIT License

---

**最近更新**: 2026-08-27 (v0.2.0)