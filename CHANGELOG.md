# Changelog

本文件记录 emd-converter 的版本变更。**版本块标题与 git tag 保持一致**：
推送 tag `v0.1.3` 时，CI 会提取 `## [v0.1.3]` 区块作为 release notes，
并自动将构建好的 Windows exe 上传为 release 资产。

格式约定（Keep a Changelog 风格）：
`### 性能 / 内存 / 可靠性 / 修复 / 工程化` 分类列表。

## [v0.3.1] - 2026-09-21

### 修复
- **并行处理死锁（打包 exe）**：GUI 入口缺少 `multiprocessing.freeze_support()`，
  Windows spawn 在冻结环境重跑 exe 时子进程会再次执行 `main()`（弹独立 GUI
  且永不回池），父进程 `as_completed` 永久等待 → 勾选"并行处理"卡死。
  源码运行不受影响（`__main__` 守卫生效）。已用同构 onefile 自测 exe 验证
  修复后 3 worker 并行正常完成（v0.3.0 已发布的 exe 存在此问题，请更新版本）
- **GUI 跨线程日志潜在死锁**：转换/并行监视线程中 `print` → `StdoutRedirector`
  直接操作 tkinter Text 属跨线程调 Tcl，高频日志下可随机死锁（表现为界面假死）。
  改为线程安全队列 + 主线程 80ms 轮询统一冲刷，日志输出顺序与体验不变

## [v0.3.0] - 2026-09-18

### 新增
- GUI 支持**拖放添加**：把 EMD 文件或文件夹直接拖到窗口任意位置即可入列
  （目录递归、去重、路径含空格/URI 形式均可解析；列表悬停高亮反馈）。
  依赖可选包 tkinterdnd2（已列入 requirements.txt；未安装时自动降级为
  按钮方式并提示，打包 exe 已通过 --collect-all 内置）
- PNG 标尺**比例化布局 + 单位进位**：新增 `_scalebar_layout`（文字带 ≈4.1%
  图高、条厚 ≈0.5% 图高，与 Velox 参考 TIFF 同阈值实测逐项对齐——0754 样本
  现输出条 19×1134px、文字带 154px、间隙 27px，与参考逐像素吻合；边距区
  从旧版 ~19% 图高收敛到 ~6%）；
  自动标尺长度改为 1/2/5×10ⁿ（旧表上限 1000 导致 "1000 nm" 原样打印），
  并经 `_normalize_scalebar_unit` 进位显示（1000 nm→1 µm、5000 nm→5 µm，
  仅改标签不改像素长度）。修复 `add_scalebar_to_axis` 厚度换算
  `/(bbox.height*dpi/72)` 将条放大约 72 倍、占满边距区的错误；
  边距区高度改按"字体 em 盒 + 条厚 + 上下间隙"自适应，字号比例下限降至
  7pt——修复 512px 元素图上文字行盒超出边距、画进图像区的问题，
  小尺寸图像的字体随之缩小
- LineProfile.png 图例从 `loc='best'`（右上角压住曲线）移到**绘图区右侧外部**
  （twinx 双轴下 best 只避开左轴数据）；外置位置在 `tight_layout` 之后按
  右轴刻度数字的实测像素边界动态右移（留 8px），避免与 twinx 刻度重叠
  （固定 1.02 锚点会压住右轴数字，如宽位数 counts 刻度）；新增 `line_profile_ylabel`，左纵轴
  标题/单位按 EDS 定量模式自动选择：AtomicFraction→"Atomic fraction (%)"、
  WeightFraction→"Weight fraction (%)"、NetIntensity→"Net intensity (counts)"、
  Intensity→"Intensity (counts)"，与 CSV 表头同一套规则；GUI/CLI/预览三处
  调用点均已传入 `parameters['quantification_mode']`
- **视角校正（Ceta 图像自动恢复屏幕显示方向）**：ImageDisplay.angle 字段实际
  以**弧度**存储显示旋转（Velox "Image Rotation" 面板读数 = degrees(angle)，
  如 ddg 2026-09 样本 angle=-4.8072409 rad ≙ 275.4° ≙ 净 84.6° CCW），
  但 EMD 存储数组从未旋转，导致导出丢失方向。现 TEM/DCFI/Crop/滤波
  导出时**全自动**按该角度旋转（90° 整数步无损，残差角双三次插值）并裁到
  最大内接方形——输出尺寸与 Velox 自身 TIFF 导出一致（0754 样本 3757²
  vs 参考 3758²，1px 为 floor/round 之差），方向相关系数 0.986。
  angle=0（2026-02 及更早批次经 Velox 界面核实确认）与 STEM/EDS/DPC/SAD
  数据恒等直通，逐字节与旧行为一致；无需任何开关或设置。
- DCFI 导出选项"仅导出最后一帧（积分图）"（GUI 新增 DCFI 选项行，默认开启）：
  DCFI 数据栈每一帧均为截至该帧的累积积分图，开启后仅将最后一帧
  （漂移校正后全部叠加的积分图像）导出为单帧 DM5/TIFF/PNG；
  原始逐帧数据仍由 TEM/STEM 分支导出。取消勾选即恢复整栈导出（旧行为），
  选项随 gui_config.json 持久化。CLI `export_dcfi_image` 菜单同步新增
  "5. 仅导出最后一帧（积分图）"
- 测试：`tests/test_dcfi_export.py`（末帧选择/整栈回归/F=1 退化，5 项）

### 修复
- CLI 导出链 `export()` 向 8 个不接收参数的 `export_*` handler 传参
  导致 TypeError（补齐 `export_type=''` 形参）
- DCFI DM5 导出引用从未赋值的 `self.dcfi_metadata`（AttributeError），
  现于 `get_dcfi_image_and_settings` 中保存解码后的元数据
- `save_image_as_png`：元数据缺失降级为 1.0/px 时比例尺长度未计算，
  随后 `None / pixel_size` 崩溃导致 PNG 完全不产出；现跳过比例尺正常出图
- **半成品文件恢复模式**：缺 `/Features` 的 Velox 写中断产物（如 Camera 46000 x
  Ceta 20250603 1413，仅 /Data/Image 落盘）不再整文件报 h5py 内部错误——
  自动扫描 `/Data/Image/*`，按 Metadata 恢复像素标定、PNG 显示范围取数据
  0.5–99.5 百分位，以 `{源文件名}-Recovered` 导出勾选的 DM5/TIFF/PNG；
  正常文件路径零影响（`parameters` 同时显式初始化，修复无特征文件的
  AttributeError）
- "空裁剪"记录（`CropFeature` 无 `imageDisplay`，仅 cropOperationPath/
  cropAnnotationPath/inputSize{0,0}，如 cichou_20260203_1051 0002 与
  Camera 1.05 Mx Ceta 20250603 1424）导致 KeyError 使**整个文件**失败；
  现跳过裁剪特征并告警，其余图像正常导出（全库预扫确认仅此一种缺键变体）

## [v0.2.0] - 2026-08-27

### 修复
- LineProfile.png 图例重叠（HAADF 与元素图例相互遮盖）

### 可靠性
- 单文件失败不中断批量，完成后汇总成功/失败清单并弹窗提示
- 元数据缺失（像素尺寸/单位）自动降级为 1.0/pixel，替代 KeyError
- 错误分类：区分文件不存在 / 损坏 / 被占用，修正误导性报错
- 确保 h5py 句柄在异常路径下释放（Windows 文件锁风险）

### 内存
- 内存自适应流式读写：低内存机器峰值从 ~2× 数据量降至 ~1.1×
  （写入侧按可用内存自动切换整块转置 / 帧组流式）
- 低内存警告与 MemoryError 专门提示（建议仅导出 DM5）

### 性能
- 可选并行处理（GUI 复选框，默认关）：ProcessPoolExecutor + spawn，
  worker 数按可用内存自适应（上限 8），大文件自动独占串行
  （14 文件实测 29s → 19.7s，1.47×；相对初始版本累计约 7.6×）

### 工程化
- pytest 测试套件（46 项）：合成最小 EMD 骨架覆盖 9 类特征，
  双线性插值数值对照、分块转置往返、DM5 写读回读、健壮性
- CI：Linux 测试门禁先于 Windows exe 构建
- 新增 CHANGELOG.md；推 tag 自动构建 + 发布 release

## [v0.1.2] - 2026-08-27

### 性能
- EMD → DM5 批量转换总耗时 150.7s → 26.6s（约 5.7×）
  - 线剖面提取向量化双线性插值（EDS Line Scan 71s → 0.4s，~70×）
  - 大数据读取适配 Velox 分块存储（逐帧读入 + 缓存分块转置，~30s → ~10s），
    并复用已打开的 h5py 句柄
  - DM5 写入缓存分块转置（确定性 ~7s，消除 ascontiguousarray 1~31s 抖动），
    删除冗余分位数计算

## [v0.1.1] - 2026-07-31

### 修复
- SAD 衍射图 PNG 比例尺正确标注倒数空间单位 (1/nm)

## [v0.1] - 2026-03-23

- 初始版本：GUI 批量转换工具（DM5 / TIFF 16-bit / PNG / CSV）
- 支持 STEM / TEM / EDS Mapping / Line Scan / DPC / DCFI / SAD