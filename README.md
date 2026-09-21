# EMD Converter

FEI Velox EMD 文件批量转换工具，支持导出为 DM5、TIFF、PNG 和 CSV 格式。

## 功能特性

- **批量处理**: 自动处理文件夹中的所有 EMD 文件
- **拖放添加**: 直接把 EMD 文件或文件夹拖入窗口（基于 tkinterdnd2，未安装时自动降级为按钮方式）
- **多格式导出**: 支持 DM5、TIFF (16-bit)、PNG、CSV 格式
- **数据类型支持**: STEM、TEM、EDS Mapping / Line Scan、DPC、DCFI、SAD 衍射图 等
- **视角校正**: Ceta 系图像（TEM/DCFI/裁剪/滤波）自动恢复 Velox 屏幕显示方向
  （显示角度以弧度存于 ImageDisplay，angle=0 的文件与 STEM/EDS 恒等直通，全自动无需设置）
- **出版级 PNG**: 底部独立边距放置比例尺（不遮挡图像），标尺长度/字号随图幅
  自适应并按 Velox 参考标定，单位自动进位（1000 nm → 1 µm）
- **高可靠容错**: 元数据缺失自动降级；"空裁剪"记录跳过不连坐；Velox 写中断的
  半成品文件按 `-Recovered` 抢救原始图像；单文件失败不中断批量并汇总报告
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

> 依赖版本提示：v0.3.0 的依赖（h5py / numpy / tifffile 等新版 wheel）已不再支持
> Python 3.8/3.9，请使用 **Python 3.10+**（本地验证 3.12，CI 验证 3.11）。

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
conda create -n emd_converter python=3.12
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

### 关于拖放依赖（tkinterdnd2）

- `requirements.txt` 已包含 `tkinterdnd2`，随上述任一方式一并安装；exe 亦已内置。
- **Linux** 需先有系统级 Tk：`sudo apt install python3-tk`（否则 GUI 本身无法启动）。
- 该依赖装不上或想移除时不影响使用：拖放功能自动降级为按钮方式，日志会提示。

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

1. 点击"选择文件夹"（或"添加文件"）按钮，选择包含 EMD 文件的文件夹；也可直接把文件/文件夹拖放到窗口上
2. 勾选需要导出的格式 (DM5、TIFF、PNG、CSV)；EDS 数据可展开 EDS 选项；DCFI 数据默认仅导出最后一帧（积分图），可取消勾选恢复整栈导出。视角校正全自动无开关；所有选项随 `gui_config.json` 持久化
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
| `tests/` | pytest 测试套件（79 项：合成 EMD 骨架覆盖 9 类特征 + 视角校正/标尺布局/拖放解析/半成品与空裁剪容错等回归） |
| `.github/workflows/build-windows-exe.yml` | 测试门禁 + Windows 单文件 exe 构建 + 自动发布流水线 |

## 支持的 EMD 数据类型

- **STEM / TEM 图像**（单张与系列）: 导出为 DM5、TIFF、PNG。Ceta 相机系图像（TEM/DCFI/裁剪/滤波）自动恢复 Velox 屏幕方向（按 ImageDisplay 显示角度旋转并裁内接方形；该字段为弧度，即 Velox "Image Rotation" 面板读数换算）；显示角度为 0 的文件与 STEM/EDS 等恒等直通，无需设置
- **EDS Mapping**（元素分布图）: 导出为 DM5、TIFF、PNG
- **EDS 能谱**（积分谱图）: 导出为 CSV、PNG 谱图
- **EDS Line Scan**: ColorMix 图像、Line Profile PNG 与 CSV
- **DPC（差分相位衬度）**: 导出为 DM5、TIFF、PNG
- **DCFI（漂移校正帧积分）**: 导出为 DM5、TIFF、PNG。DCFI 数据栈的每一帧均为"截至该帧的累积积分图"，默认仅导出最后一帧（全部叠加后的积分图像）；原始逐帧数据由 TEM/STEM 分支导出。取消勾选"仅导出最后一帧"可恢复导出整个累积序列
- **SAD 衍射图**: 导出为 DM5、TIFF、PNG（比例尺正确标注 1/nm 倒数空间单位）

## 输出文件命名规则

转换后的文件保存在 `输出目录/{源文件名}/` 子文件夹中，按数据类型命名：

| 数据 | 命名 | 示例 |
|---|---|---|
| TEM（Ceta） | `{源文件名}.{格式}` | `sample_0754_Camera_Ceta.dm5` |
| STEM / DPC 探测器 | `{源文件名}-{探测器}.{格式}` | `sample_STEM-HAADF.tif` |
| DCFI | `{Velox 显示标签} of {源文件名}.{格式}`（默认仅末帧积分图） | `DCFI(Ceta) of sample_0802.png` |
| EDS 元素图 / HAADF | `{源文件名}-{元素}-{定量模式}.{格式}` | `sample_SI-Al-AtomicFraction.tif` |
| Color Mix / Line Profile | `{源文件名}-Colormix[-LineAnnotation].png`、`{源文件名}-LineProfile.{png,csv}` | — |
| 定量结果 | `{源文件名}-Quantification-{模式}.csv` | — |
| 半成品文件恢复 | `{源文件名}-Recovered[-NN].{格式}` | `Camera 46000…-Recovered.dm5` |

## 注意事项

- TIFF 导出使用 16-bit 格式以保留完整动态范围
- 像素尺寸、单位等校准信息会保留在 TIFF 和 DM5 文件的元数据中
- 大文件（数 GB 级原位序列）处理需要较多内存；低内存机器会自动切换流式模式并给出提示
- Velox 写中断产生的"半成品"文件（缺 /Features）会以 `-Recovered` 命名尽力抢救 `/Data/Image` 中的原始图像；恢复结果缺显示旋转/窗口布局信息，属预期
- "空裁剪"记录（只有裁剪操作、没有裁剪结果图像的 CropFeature）自动跳过并告警，不影响该文件其余图像导出
- 视角校正对像素做任意角重采样（90° 整数步无损；残差角双三次插值）并裁最大内接方形，产物方向与 Velox 自身 TIFF 导出一致，但与文件存储数组不同；若需严格未旋转的存储像素（如对接旧流程），请使用 v0.2.0 或更早版本导出
- 勾选"并行处理"可加速多文件批次（默认关闭；大文件按内存预算自动独占串行）
- 配置文件 `gui_config.json` 会自动保存用户设置（格式、EDS、DCFI 末帧等选项状态）

## 系统要求

- Python 3.10+（推荐 3.12；CI 验证 3.11）
- Windows / Linux / macOS（Linux 需系统 tkinter：`python3-tk`）

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

可选依赖：
- tkinterdnd2 >= 0.4.0（GUI 拖放添加文件/文件夹；缺失时功能自动降级，不影响其余功能）

## 许可证

MIT License

---

**最近更新**: 2026-09-18 (v0.3.0)