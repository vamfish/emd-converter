# Changelog

本文件记录 emd-converter 的版本变更。**版本块标题与 git tag 保持一致**：
推送 tag `v0.1.3` 时，CI 会提取 `## [v0.1.3]` 区块作为 release notes，
并自动将构建好的 Windows exe 上传为 release 资产。

格式约定（Keep a Changelog 风格）：
`### 性能 / 内存 / 可靠性 / 修复 / 工程化` 分类列表。

## [v0.1.3] - 2026-08-27

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
  （14 文件实测 29s → 19.7s，1.47×）

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