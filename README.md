# EMD Converter

FEI Velox EMD 文件批量转换工具，支持导出为 DM5、TIFF、PNG 和 CSV 格式。

## 功能特性

- **批量处理**: 自动处理文件夹中的所有 EMD 文件
- **多格式导出**: 支持 DM5、TIFF (16-bit)、PNG、CSV 格式
- **数据类型支持**: STEM、TEM、EDS、DPC、DCFI 等
- **元数据保留**: 保留像素尺寸、单位等信息
- **用户友好**: 图形界面，支持配置自动保存

## 安装

### 方法 1: 使用 pip

```bash
pip install -r requirements.txt
```

### 方法 2: 使用 Conda (推荐)

```bash
conda create -n emd_converter python=3.10
conda activate emd_converter
pip install -r requirements.txt
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

1. 点击"选择文件夹"按钮，选择包含 EMD 文件的文件夹
2. 勾选需要导出的格式 (DM5、TIFF、PNG、CSV)
3. 点击"开始处理"
4. 转换后的文件将保存在 `emd_converted` 子文件夹中

## 文件说明

| 文件 | 大小 | 说明 |
|------|------|------|
| `emd_converter_gui.py` | ~40 KB | 主程序，图形界面 |
| `velox_file_analyzer2.py` | ~170 KB | EMD 文件解析核心库 |
| `launch_gui.py` | ~1 KB | 启动脚本 |
| `requirements.txt` | ~250 B | Python 依赖列表 |
| `README.md` | ~2 KB | 说明文档 |

## 支持的 EMD 数据类型

- **STEM/TEM 图像**: 导出为 DM5、TIFF、PNG
- **EDS 能谱**: 导出为 CSV、PNG 谱图
- **DPC (差分相位衬度)**: 导出为 DM5、TIFF、PNG
- **DCFI (漂移校正帧积分)**: 导出为 DM5、TIFF、PNG
- **线扫描 (Line Scan)**: 导出为 CSV

## 输出文件命名规则

转换后的文件保存在 `emd_converted` 文件夹中，命名格式：
- `{原文件名}_{数据类型}_{编号}.{格式}`

例如：`sample_DF_0001.dm5`

## 注意事项

- TIFF 导出使用 16-bit 格式以保留完整动态范围
- 像素尺寸信息会保留在 TIFF 和 DM5 文件的元数据中
- 大文件处理可能需要一些时间，请耐心等待
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

**项目精简日期**: 2026-03-23
