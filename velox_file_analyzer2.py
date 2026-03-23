"""
Velox 文件分析器 - 用于解析和处理 Velox EMD 文件中的数据。
包含图像、谱图、线扫描、颜色混合等多种数据处理功能。
"""

import os
import re
import json
import math
import csv
import tifffile
import h5py
import numpy as np

# 设置 Matplotlib 使用非交互式后端（必须在导入 pyplot 之前）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免多线程问题

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrow
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from PIL import Image
from pathlib import Path
from urllib.parse import unquote
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline
from bs4 import BeautifulSoup
from typing import Optional, Tuple, Dict, List, Any, Union

# 设置全局样式
plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei'],
    'axes.unicode_minus': False,
    'figure.figsize': (10, 8),
    'figure.dpi': 100,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
})
# ============================================================================
# 辅助函数
# ============================================================================

def bytes_to_json(h5dataset, need_print=False) -> Dict[str, Any]:
    """
    将 shape (1,), type "|O" 的 HDF5 dataset 转换为 JSON 格式的字典。
    
    参数:
        h5dataset: h5py.Dataset 或 numpy.ndarray 对象
        need_print: 是否打印结果
    
    返回:
        JSON 字典
        
    异常:
        TypeError: 输入类型不匹配
        ValueError: 形状或数据类型不匹配
    """
    # 验证输入类型
    if not isinstance(h5dataset, (h5py.Dataset, np.ndarray)):
        raise TypeError(
            f"类型不匹配：期望 h5py.Dataset 或 numpy.ndarray，实际 {type(h5dataset)}"
        )
    
    # 验证形状
    if h5dataset.shape != (1,):
        raise ValueError(f"形状不匹配：期望 (1,)，实际 {h5dataset.shape}")
    
    # 验证数据类型
    if h5dataset.dtype != 'object':
        raise ValueError(f"数据类型不匹配：期望 object，实际 {h5dataset.dtype}")
    
    # 解码 JSON
    json_dict = json.loads(h5dataset[0].decode('utf-8'))
    
    if need_print:
        print(json.dumps(json_dict, indent=4, ensure_ascii=False))
    
    return json_dict

def decode_metadata(metadata, need_print=False) -> Dict[str, Any]:
    """
    将 metadata 从 uint8 数组恢复为字符串并解析为 JSON。
    
    参数:
        metadata: h5py.Dataset 或 numpy.ndarray 对象
        need_print: 是否打印结果
    
    返回:
        JSON 字典
        
    异常:
        TypeError: 输入类型不匹配
        ValueError: 形状、数据类型或数值越界
    """
    # 验证输入类型
    if not isinstance(metadata, (h5py.Dataset, np.ndarray)):
        raise TypeError(
            f"类型不匹配：期望 h5py.Dataset 或 numpy.ndarray，实际 {type(metadata)}"
        )
    
    # 验证形状
    if metadata.shape[0] != 60000:
        raise ValueError(f"形状不匹配：期望 (60000, int)，实际 {metadata.shape}")
    
    # 验证数据类型
    if metadata.dtype != 'uint8':
        raise ValueError(f"数据类型不匹配：期望 uint8，实际 {metadata.dtype}")
    
    # 验证数值范围
    if metadata[:, 0].min() < 0 or metadata[:, 0].max() > 127:
        raise ValueError("数值越界：所有元素必须落在 0–127 范围内")
    
    # 解码元数据
    metadata_json = json.loads(
        metadata[:, -1].ravel()
        .tobytes()
        .decode('ascii')
        .strip('\x00')
        .strip()
    )
    
    if need_print:
        print(json.dumps(metadata_json, indent=4))
    
    return metadata_json

def html_table_to_csv(
    html_content: str,
    output_path: Optional[str] = None,
    table_index: int = 0,
    encoding: str = 'utf-8',
    include_header: bool = True,
    delimiter: str = ',',
    quotechar: str = '"',
    add_metadata: bool = False,
    metadata: Optional[dict] = None
) -> Union[str, None]:
    """
    将 HTML 表格转换为 CSV 格式。
    
    参数:
        html_content: 包含 HTML 表格的字符串
        output_path: 输出 CSV 文件的路径，如果为 None 则返回 CSV 字符串
        table_index: 要提取的表格索引（当有多个表格时）
        encoding: 文件编码
        include_header: 是否包含表头
        delimiter: CSV 分隔符
        quotechar: CSV 引用符
        add_metadata: 是否在 CSV 开头添加元数据
        metadata: 要添加的元数据字典
    
    返回:
        如果 output_path 为 None，返回 CSV 字符串；否则返回 None，文件保存到指定路径
        
    异常:
        ValueError: HTML 内容中没有找到表格
        RuntimeError: 转换失败
    """
    try:
        # 解析 HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        if not tables:
            raise ValueError("HTML 内容中没有找到表格")
        
        if table_index >= len(tables):
            raise ValueError(f"表格索引 {table_index} 超出范围，共找到 {len(tables)} 个表格")
        
        table = tables[table_index]
        
        # 提取表头
        headers = []
        thead = table.find('thead')
        
        if thead:
            # 从 thead 提取表头
            header_rows = thead.find_all('tr')
            for row in header_rows:
                cells = row.find_all(['th', 'td'])
                if cells:
                    row_headers = [cell.get_text(strip=True) for cell in cells]
                    headers.extend(row_headers)
        else:
            # 如果没有 thead，尝试从第一行提取
            first_row = table.find('tr')
            if first_row:
                th_cells = first_row.find_all('th')
                if th_cells:
                    headers = [cell.get_text(strip=True) for cell in th_cells]
        
        # 提取数据行
        data_rows = []
        all_rows = table.find_all('tr')
        
        # 确定起始行
        start_index = 0
        if thead:
            start_index = len(thead.find_all('tr'))
        elif headers and len(all_rows) > 0:
            first_row = all_rows[0]
            if first_row.find_all('th'):
                start_index = 1
        
        # 处理数据行
        for i in range(start_index, len(all_rows)):
            row = all_rows[i]
            cells = row.find_all(['td', 'th'])
            
            if cells:
                row_data = [cell.get_text(strip=True) for cell in cells]
                data_rows.append(row_data)
        
        # 如果还没有表头，检查第一行数据是否可能是表头
        if not headers and data_rows:
            first_data_row = data_rows[0]
            is_header_row = all(
                not re.match(r'^-?\d*\.?\d+$', cell) and cell
                for cell in first_data_row
            )
            if is_header_row and include_header:
                headers = first_data_row
                data_rows = data_rows[1:]  # 移除表头行
        
        # 创建 CSV 内容
        output = []
        
        # 添加元数据
        if add_metadata:
            if metadata:
                for key, value in metadata.items():
                    output.append([f"# {key}: {value}"])
            else:
                import datetime
                output.append([f"# 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
                output.append([f"# 表格索引: {table_index}"])
                output.append([f"# 数据行数: {len(data_rows)}"])
                output.append([f"# 列数: {len(headers) if headers else len(data_rows[0]) if data_rows else 0}"])
                output.append([])  # 空行分隔
        
        # 添加表头
        if include_header and headers:
            output.append(headers)
        
        # 添加数据
        output.extend(data_rows)
        
        # 写入文件或返回字符串
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', newline='', encoding=encoding) as csvfile:
                writer = csv.writer(
                    csvfile,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    quoting=csv.QUOTE_MINIMAL
                )
                writer.writerows(output)
            
            print(f"CSV 文件已保存到: {output_path}")
            print(f"列数: {len(headers) if headers else '无表头'}")
            print(f"数据行数: {len(data_rows)}")
            return None
        else:
            import io
            with io.StringIO() as string_buffer:
                writer = csv.writer(
                    string_buffer,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    quoting=csv.QUOTE_MINIMAL
                )
                writer.writerows(output)
                csv_string = string_buffer.getvalue()
            return csv_string
            
    except Exception as e:
        raise RuntimeError(f"转换失败: {str(e)}")

def export_eds_spectrum(output_path, intensity, offset=-250, channels=4096, dispension=5):
    """
    将 eds spectrum 写入 csv
    参数：
        output_path: 输出路径
        intensity: eds 数据，一般是4096的1D array
        offset: 起始能量
        channels: 通道数
        dispension: 每个通道的能量间隔
    """
    if intensity.ndim != 1 or intensity.size != channels:
        raise ValueError(f"intensity 必须是长度为 {channels} 的一维数组")
    energy = np.arange(channels, dtype=float) * dispension + offset + dispension / 2
    header1 = "Energy (eV)"
    header2 = "Intensity (Counts)"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([header1, header2])
        for val1, val2 in zip(energy, intensity):
            writer.writerow([val1, val2])
    print(f"成功保存到: {output_path}")

def export_line_profile_as_csv(output_path, line_length_pixel, line_profile, pixelsize, pixelunit, quantification_mode):
    """
    将 line profile 写入 csv
    参数：
        output_path: 输出路径
        line_length_pixel: line profile 的像素长度，self.line_position['length']
        line_profile: line profile，dict[element]['profile_avg']
        pixelsize: 像素尺寸
        pixelunit: 像素单位
        quantification_mode: 定量模式
    """
    if not isinstance(line_profile, dict):
        raise TypeError(f"line_profile不是dict，而是{type(line_profile)}")
    line_datas = {}
    for element, value in line_profile.items():
        try:
            line_datas[element] = value['profile_avg']
            data_length = line_datas[element].shape[0]
        except Exception as e:
            raise ValueError(f"line_profile没有profile_avg：{e}")
            return None
    headrow1 = ["",]
    headrow2 = ["Position",]
    headrow3 = [pixelunit,]
    for element in line_datas.keys():
        headrow1.append(element)
        if 'DF' in element or 'BF' in element:
            headrow2.append("Intensity")
            headrow3.append("Counts")
        else:
            headrow2.append(quantification_mode)
            if 'NetIntensity' in quantification_mode:
                headrow3.append("Counts")
            elif 'WeightFraction' in quantification_mode:
                headrow3.append("wt.%")
            elif 'AtomicFraction' in quantification_mode:
                headrow3.append("at.%")
    positions = np.arange(data_length, dtype=float) * pixelsize * line_length_pixel / data_length
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for headrow in (headrow1, headrow2, headrow3):
            writer.writerow(headrow)
        for index in range(data_length):
            datarow = []
            datarow.append(positions[index])
            for element in line_datas.keys():
                datarow.append(line_datas[element][index])
            writer.writerow(datarow)
    print(f"成功保存到: {output_path}")
    return output_path

def optimized_read_with_progress(file_path, dataset_name="/Data/Image/ad5b0bc2b43d40dfb366f41a2b3839d5/Data"):
    """
    优化的 HDF5 数据读取策略，带进度显示。
    
    参数:
        file_path: HDF5 文件路径
        dataset_name: 数据集路径
    
    返回:
        读取的数据数组
    """
    with h5py.File(file_path, 'r') as f:
        dataset = f[dataset_name]
        
        # 获取存储信息
        if dataset.chunks:
            print(f"数据集使用分块存储，块大小: {dataset.chunks}")
        
        if dataset.compression:
            print(f"数据集使用 {dataset.compression} 压缩")
        
        shape = dataset.shape
        print(f"数据形状: {shape}, 数据类型: {dataset.dtype}")
        
        # 根据维度选择读取策略
        if len(shape) == 3:
            # 3D 数据按 Z 轴切片读取
            result = np.empty(shape, dtype=dataset.dtype)
            
            for z in tqdm(
                range(shape[2]),
                desc=f"读取 {dataset_name}",
                unit='slice',
                mininterval=0.5
            ):
                result[:, :, z] = dataset[:, :, z]
                
        elif len(shape) == 2:
            # 2D 数据按行读取
            result = np.empty(shape, dtype=dataset.dtype)
            
            for y in tqdm(
                range(shape[0]),
                desc=f"读取 {dataset_name}",
                unit='row'
            ):
                result[y, :] = dataset[y, :]
        else:
            # 小数据直接读取
            result = dataset[:]
        
        return result

def display_two_grayscale_images(
    image1: np.ndarray,
    image2: np.ndarray,
    display_range1: Optional[Tuple[float, float]] = None,
    display_range2: Optional[Tuple[float, float]] = None,
    titles: Tuple[str, str] = ("Image 1", "Image 2"),
    figsize: Tuple[int, int] = (15, 8),
    cmap: str = 'gray',
    share_axes: bool = False,
    show_colorbar: bool = False,
    **kwargs
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    同时显示两幅灰度图像，每幅图像可以自定义显示范围。
    
    参数:
        image1, image2: 要显示的两幅图像，形状为 (H, W)
        display_range1, display_range2: 每幅图像的显示范围 (vmin, vmax)
        titles: 每幅图像的标题
        figsize: 图形大小 (width, height) in inches
        cmap: 颜色映射
        share_axes: 是否共享坐标轴范围
        show_colorbar: 是否显示颜色条
        **kwargs: 传递给 imshow 的其他参数
    
    返回:
        fig: 图形对象
        axes: 两个坐标轴对象
    """
    # 参数验证
    if image1.shape != image2.shape:
        print(f"警告: 图像形状不同 - image1: {image1.shape}, image2: {image2.shape}")
    
    # 创建图形和坐标轴
    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        sharex=share_axes,
        sharey=share_axes
    )
    
    # 设置显示范围
    vmin1, vmax1 = (display_range1 if display_range1 is not None
                   else (np.nanmin(image1), np.nanmax(image1)))
    
    vmin2, vmax2 = (display_range2 if display_range2 is not None
                   else (np.nanmin(image2), np.nanmax(image2)))
    
    # 显示第一幅图像
    im1 = axes[0].imshow(
        image1,
        vmin=vmin1,
        vmax=vmax1,
        cmap=cmap,
        aspect='auto',
        **kwargs
    )
    axes[0].set_title(f"{titles[0]}\nRange: [{vmin1:.2f}, {vmax1:.2f}]", fontsize=12)
    axes[0].axis('off')
    
    # 显示第二幅图像
    im2 = axes[1].imshow(
        image2,
        vmin=vmin2,
        vmax=vmax2,
        cmap=cmap,
        aspect='auto',
        **kwargs
    )
    axes[1].set_title(f"{titles[1]}\nRange: [{vmin2:.2f}, {vmax2:.2f}]", fontsize=12)
    axes[1].axis('off')
    
    # 添加颜色条
    if show_colorbar:
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 调整布局
    plt.tight_layout()
    
    return fig, axes

def display_image_with_scale(
    image: np.ndarray,
    pixel_size: float = 1.0,
    pixel_unit: str = 'pixel',
    title: str = 'Image',
    cmap: str = 'gray',
    display_range: tuple = None,
    gamma: float = 1.0,
    display_index: int = 0,
    show_colorbar: bool = True,
    aspect: str = 'equal',
    origin: str = 'upper',
    figsize: tuple = (10, 8),
    dpi: int = 100
) -> tuple:
    """
    简洁的带物理尺寸图像展示函数
    
    参数:
        image: 2D/3D numpy数组
        pixel_size: 像素尺寸 (x和y方向相同)
        pixel_unit: 物理尺寸单位
        title: 标题
        cmap: 颜色映射
        display_range: (vmin, vmax) 图像显示范围
        gamma: Gamma 校正值，1.0 表示无校正
        display_index: 3D 图像的展示 slice index
        show_colorbar: 是否显示颜色条
        aspect: 显示比例，'equal'保持1:1
        origin: 原点位置，'upper'表示左上角为原点
        figsize: 图形大小
        dpi: 分辨率
    
    返回:
        fig, ax: 图形和坐标轴对象
    """
    # 计算物理尺寸
    if len(image.shape) == 3:
        image = image[:,:,display_index]
    height, width = image.shape
    phys_width = width * pixel_size
    phys_height = height * pixel_size
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 设置显示范围
    if display_range:
        vmin = np.min(display_range)
        vmax = np.max(display_range)
    else:
        vmin, vmax = np.min(image), np.max(image)
    
    # 使用 PowerNorm 进行 Gamma 校正
    if gamma != 1.0:
        # 使用 PowerNorm 进行 Gamma 校正
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    else:
        norm = None  # 不使用 Gamma 校正
    
    # 显示图像（使用物理坐标范围）
    # extent格式: [left, right, bottom, top]
    # 左上角为原点时，top=0, bottom=phys_height
    extent = [0, phys_width, phys_height, 0]
    
    im = ax.imshow(
        image,
        cmap=cmap,
        norm=norm,
        vmin=vmin if norm is None else None,  # 如果用了 norm，就不需要 vmin/vmax
        vmax=vmax if norm is None else None,
        origin=origin,
        aspect=aspect,
        extent=extent,
        interpolation='nearest'
    )
    
    # 设置标题和坐标轴
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    if pixel_unit != 'pixel':
        ax.set_xlabel(f'X ({pixel_unit})', fontsize=12)
        ax.set_ylabel(f'Y ({pixel_unit})', fontsize=12)
    else:
        ax.set_xlabel('X (pixel)', fontsize=12)
        ax.set_ylabel('Y (pixel)', fontsize=12)
    
    # 显示颜色条
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Intensity', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    return fig, ax

def plot_spectrum(
        counts: np.ndarray,
        *,
        start: float = -247.5,
        end: float = 20227.5,
        log: bool = False,
        title: str = "Energy Spectrum",
        xlabel: str = "Energy (eV)",
        ylabel: str = "Counts",
        figsize: Tuple[int, int] = (7, 4),
        save_path: Optional[str] = None,
        dpi: int = 300,
        show: bool = False,
        grid: bool = True,
        minor_ticks: bool = True,
        color: str = "steelblue",
        lw: float = 0.8,
) -> plt.Figure:
    """
    快速绘制 4096 通道能谱图（支持任意通道→能量线性标定）。

    Parameters
    ----------
    counts : ndarray, shape=(4096,)
        每通道计数，dtype 不限，会内部转 float。
    start : float
        通道 0 对应的能量，单位 eV。
    end : float
        通道 4095 对应的能量，单位 eV。
    log : bool
        True 则纵轴对数坐标；若含 0 会自动把 0 替换为 0.1。
    save_path : str, optional
        若给出，则保存图片（支持 pdf/png/svg…）。
    show : bool
        是否立即 plt.show()；设为 False 可继续在外部二次加工。
   其余参数见名知意。

    Returns
    -------
    fig : matplotlib.figure.Figure
        方便外部继续装饰或 savefig。
    """
    if counts.ndim != 1 or counts.size != 4096:
        raise ValueError("counts 必须是长度为 4096 的一维数组")
    # 1. 能量标定：线性映射
    k = (end - start) / 4095
    energy = np.arange(4096, dtype=float) * k + start   # eV
    # 2. 转 float，处理对数 0 问题
    y = counts.astype(float)
    if log:
        y = np.where(y <= 0, 0.1, y)
    # 3. 画图
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(energy, y, lw=lw, color=color)
    # 4. 坐标轴
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # 5. 网格与副刻度
    if minor_ticks:
        ax.minorticks_on()
    if grid:
        ax.grid(which="major", ls="-", alpha=0.4)
        ax.grid(which="minor", ls=":", alpha=0.2)
    # 6. 保存
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    # 7. 显示
    if show:
        plt.show()
    return fig

def draw_line_annotation_on_image(color_image, line_info=None, pixel_size=1.0, pixel_unit='pixel'):
    """
    第一幅图：在彩色图像上标注线位置
    
    参数:
        color_image: 彩色图像，形状 (H, W, 3)
        line_info: 线条信息字典，包含 'start', 'end', 'line_width'
        pixel_size: 像素尺寸
        pixel_unit: 物理尺寸单位
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 计算物理尺寸
    height, width = color_image.shape[:2]
    phys_width = width * pixel_size
    phys_height = height * pixel_size
    
    # 显示彩色图像
    extent = [0, phys_width, phys_height, 0]  # 左上角为原点
    ax.imshow(color_image, origin='upper', aspect='equal', extent=extent)
    
    # 提取线条信息
    if line_info:
        start_px = line_info['start']  # (x, y) 像素坐标
        end_px = line_info['end']      # (x, y) 像素坐标
        line_width_px = line_info['line_width']
    
        # 转换为物理坐标
        start_phys = (start_px[0] * pixel_size, start_px[1] * pixel_size)
        end_phys = (end_px[0] * pixel_size, end_px[1] * pixel_size)
    
        # 计算线宽的一半（用于绘制方框）
        half_width = line_width_px * pixel_size / 2
    
        # 计算线条方向向量和垂直向量
        dx = end_phys[0] - start_phys[0]
        dy = end_phys[1] - start_phys[1]
        line_length = np.sqrt(dx**2 + dy**2)
    
        # 归一化方向向量
        if line_length > 0:
            ux = dx / line_length
            uy = dy / line_length
        else:
            ux, uy = 1, 0
        
        # 垂直向量（逆时针旋转90度）
        vx = -uy
        vy = ux
        
        # 计算矩形（方框）的四个角点
        # 矩形沿着线条方向，宽度为line_width
        p1 = (start_phys[0] + vx * half_width, start_phys[1] + vy * half_width)
        p2 = (start_phys[0] - vx * half_width, start_phys[1] - vy * half_width)
        p3 = (end_phys[0] - vx * half_width, end_phys[1] - vy * half_width)
        p4 = (end_phys[0] + vx * half_width, end_phys[1] + vy * half_width)
        
        # 绘制矩形（方框表示线宽）
        rect_polygon = Polygon([p1, p2, p3, p4], closed=True, 
                               fill=False, edgecolor='darkred', linewidth=2)
        ax.add_patch(rect_polygon)

        # 绘制箭头
        ax.annotate('', xy=end_phys, xytext=start_phys, arrowprops=dict(arrowstyle='->', color='yellow', linewidth=2, shrinkA=0, shrinkB=0), zorder=5)

        ax.set_title(f'Line Annotation on Image\nLine Width: {line_width_px} pixels', fontsize=14, fontweight='bold', pad=20)

    # 设置坐标轴和标题
    ax.set_xlabel(f'X ({pixel_unit})', fontsize=12)
    ax.set_ylabel(f'Y ({pixel_unit})', fontsize=12)

    plt.tight_layout()
    return fig, ax

def draw_line_profiles(profiles_data, output_path: str = None, pixel_size=1.0, pixel_unit='pixel', line_length_px=None, figsize=(12, 8)):
    """
    第二幅图：绘制多条线剖面图
    
    参数:
        profiles_data: 线剖面数据列表，每个元素是一个一维数组
        output_path: 输出PNG文件路径
        pixel_size: 像素尺寸
        pixel_unit: 物理尺寸单位
        profile_names: 每条剖面的名称列表
        line_length_px: 线的像素长度（用于计算物理距离）
        figsize: 图形大小
    """
    profiles = []
    colors = []
    names = []
    for key, value in profiles_data.items():
        names.append(key)
        profiles.append(value['profile_avg'])
        color = (value['color']['red'], value['color']['green'], value['color']['blue'])
        if color == (1, 1, 1):
            color = (0, 0, 0)
        colors.append(color)
    # 创建图形
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()
    
    # 检查数据
    if not profiles:
        print("错误: 没有提供剖面数据")
        return fig, ax
    
    # 计算横坐标（物理距离）
    # 假设每条剖面的采样点数相同，且在线段上等距采样
    right_lines = []
    left_lines = []
    for i, profile in enumerate(profiles):
        n_points = len(profile)
        
        # 计算物理距离
        if line_length_px is not None:
            # 如果有线的像素长度，计算物理长度
            line_length_phys = line_length_px * pixel_size
            x_physical = np.linspace(0, line_length_phys, n_points)
        else:
            # 否则使用像素索引
            x_physical = np.arange(n_points) * pixel_size
        
        if re.search(r'DF|BF', names[i], re.IGNORECASE):
            line = ax_right.plot(x_physical, profile, 
                    color=colors[i], 
                    linewidth=2,  # 使用虚线区分
                    label=names[i])[0]
            right_lines.append(line)
        else:
            line = ax_left.plot(x_physical, profile, 
                    color=colors[i], 
                    linewidth=2,  # 使用虚线区分
                    label=names[i])[0]
            left_lines.append(line)
    
    # 设置坐标轴和标题
    ax_left.set_ylabel('Intensity of Elements', fontsize=12, color='blue')
    ax_left.tick_params(axis='y', labelcolor='blue')
    ax_right.set_ylabel('Intensity of STEM Image', fontsize=12, color='red')
    ax_right.tick_params(axis='y', labelcolor='red')
    ax_left.set_xlabel(f'Distance along line ({pixel_unit})', fontsize=12)
    ax_left.set_title('Line Profiles', fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格
    ax_left.grid(True, alpha=0.3)
    
    # 添加图例
    ax_left.legend(loc='best', fontsize=10)
    ax_right.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        # 处理输出路径
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # 保存为PNG
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0,
            format='png'
        )

    return fig

def save_as_16bit_tiff(data, output_path, metadata={}):
    """
    保存TIFF并包含转换参数，便于后期恢复
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("输入数据必须是numpy数组")
    # if len(data.shape) != 2:
        # raise ValueError(f"输入数据必须是2D数组，当前形状为 {data.shape}")
    if data.dtype == 'float32':
        uint16_data = convert_float32_to_uint16(data, method = 'percentile')
    elif data.dtype == 'int16':
        uint16_data = convert_int16_to_uint16(data)
    elif data.dtype == 'uint16':
        uint16_data = data
    pixels_per_unit = 1.0 / metadata['spacing']
    metadata.update({
        'original_dtype': data.dtype,
        'original_min': float(data.min()),
        'original_max': float(data.max()),
        'scaling_method': 'linear',
        'bit_depth': 16
    })
    
    # 保存TIFF（可以嵌入元数据）
    # 正确处理文件名：传入的 output_path 已经是不带后缀的路径，直接添加 .tiff
    name = output_path.name
    # 只有当文件名以 .tiff 结尾时才保持不变，否则添加 .tiff
    if not name.lower().endswith('.tiff'):
        target_path = output_path.parent / (name + '.tiff')
    else:
        target_path = output_path
    # 处理文件已存在的情况
    if target_path.exists():
        print(f"\n文件已存在: {target_path}")
        print("请选择操作:")
        print("1. 覆盖")
        print("2. 重命名")
        print("3. 取消")
        while True:
            choice = input("请输入选择 (1, 2, 3): ").strip()
            if choice == '1':
                print("将覆盖现有文件...")
                break
            elif choice == '2':
                # 生成新文件名
                new_path = generate_unique_filename(target_path)
                print(f"将保存为: {new_path}")
                target_path = new_path
                break
            elif choice == '3':
                print("操作已取消")
                return None
            else:
                print("无效选择，请重新输入")
    # 保存TIFF文件
    if uint16_data.ndim == 3:
        # 3D 数据: 将 (H, W, C) 转换为 (C, H, W) 以符合 ImageJ 格式
        uint16_data = uint16_data.transpose(2, 0, 1)
        # 对于多帧图像，设置正确的 slices 和 frames
        if 'slices' not in metadata and uint16_data.shape[0] > 1:
            metadata['slices'] = uint16_data.shape[0]
    
    # 保存TIFF (使用原始的 pixels_per_unit，不转换单位)
    # 注意：ImageJ 格式通常使用 pixel/cm，但这里保持原始计算
    resolution = (pixels_per_unit, pixels_per_unit)
    
    tifffile.imwrite(
        str(target_path),
        uint16_data,
        imagej=True,
        photometric='minisblack',  # 灰度图像
        metadata=metadata,
        resolution=resolution,
        compression=None  # 无压缩
    )
    print(f"成功保存: {target_path}")
    
    return str(target_path)

def generate_unique_filename(file_path):
    """
    生成唯一的文件名，通过添加数字后缀
    
    Args:
        file_path (Path): 原始文件路径
    
    Returns:
        Path: 新的唯一文件路径
    """
    counter = 2
    
    # 正确处理包含多个点的文件名
    name = file_path.name
    if '.' in name:
        # 分离文件名和扩展名（从最右侧的 . 分割）
        original_stem, original_suffix = name.rsplit('.', 1)
        original_suffix = '.' + original_suffix
    else:
        original_stem = name
        original_suffix = ''
    
    while True:
        # 移除可能已存在的数字后缀
        if original_stem.endswith(f"({counter-1})"):
            original_stem = original_stem[:-(len(f"({counter-1})"))]
        
        # 构建新文件名
        new_stem = f"{original_stem}({counter})"
        new_path = file_path.parent / f"{new_stem}{original_suffix}"
        
        if not new_path.exists():
            return new_path
        
        counter += 1

def add_suffix_safe(output_path: Path, suffix: str) -> Path:
    """
    安全地添加文件后缀，正确处理包含多个点的文件名
    
    Args:
        output_path: 原始文件路径（可能包含或不包含后缀）
        suffix: 要添加的后缀（如 '.dm5', '.png'）
    
    Returns:
        Path: 添加后缀后的新路径
    """
    # 确保后缀以点开头
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    
    # 获取文件名
    name = output_path.name
    
    # 如果文件名已经以该后缀结尾，直接返回
    if name.lower().endswith(suffix.lower()):
        return output_path
    
    # 直接添加后缀（传入的 output_path 通常已经是不带后缀的文件名）
    return output_path.parent / (name + suffix)

def convert_float32_to_uint16(data, method='percentile'):
    """
    将float32数据转换为uint16
    
    Args:
        data: float32 numpy数组
        method: 转换方法
    
    Returns:
        uint16 numpy数组
    """
    if method == 'stretch':
        # 线性拉伸到0-65535
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val > min_val:
            normalized = (data - min_val) / (max_val - min_val)
            uint16_data = (normalized * 65535).astype(np.uint16)
        else:
            uint16_data = np.zeros_like(data, dtype=np.uint16)
            
    elif method == 'percentile':
        # 使用百分位裁剪，避免异常值
        p_low, p_high = np.percentile(data, [0.5, 99.5])  # 使用0.5%和99.5%分位数
        
        # 如果分位数太接近，使用实际最小最大值
        if (p_high - p_low) < (np.max(data) - np.min(data)) * 0.01:
            p_low, p_high = np.min(data), np.max(data)
        
        # 裁剪和归一化
        clipped = np.clip(data, p_low, p_high)
        normalized = (clipped - p_low) / (p_high - p_low)
        uint16_data = (normalized * 65535).astype(np.uint16)
        
    elif method == 'direct':
        # 假设数据已经在0-1范围内
        uint16_data = np.clip(data * 65535, 0, 65535).astype(np.uint16)
        
    else:
        raise ValueError(f"未知的转换方法: {method}")
    
    return uint16_data
    
def convert_int16_to_uint16(data, method='direct'):
    """
    将int16数据转换为uint16
    
    Args:
        data: int16 numpy数组
        method: 转换方法
    
    Returns:
        uint16 numpy数组
    """
    if method == 'stretch':
        # 线性拉伸到0-65535
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val > min_val:
            normalized = (data - min_val) / (max_val - min_val)
            uint16_data = (normalized * 65535).astype(np.uint16)
        else:
            uint16_data = np.zeros_like(data, dtype=np.uint16)
            
    elif method == 'percentile':
        # 使用百分位裁剪，避免异常值
        p_low, p_high = np.percentile(data, [0.5, 99.5])  # 使用0.5%和99.5%分位数
        
        # 如果分位数太接近，使用实际最小最大值
        if (p_high - p_low) < (np.max(data) - np.min(data)) * 0.01:
            p_low, p_high = np.min(data), np.max(data)
        
        # 裁剪和归一化
        clipped = np.clip(data, p_low, p_high)
        normalized = (clipped - p_low) / (p_high - p_low)
        uint16_data = (normalized * 65535).astype(np.uint16)
        
    elif method == 'direct':
        # 数据的上下限相差不超过 65535
        data_range = data.max() - data.min()
        if data_range <= 65535:
            if data.min() >= 0:
                uint16_data = data.astype(np.uint16)
            else:
                data -= data.min()
                uint16_data = data.astype(np.uint16)
        else:
            print(f"data 的取值范围 {data_range} > 65535，无法使用 direct 方法，改为线性拉伸")
            uint16_data = convert_int16_to_uint16(data, method='stretch')
        
    else:
        raise ValueError(f"未知的转换方法: {method}")
    
    return uint16_data

def dm5_writer(filename, signal, parameters):
    """
    已经适配了 SI，后续需要适配其他类型数据@20260121
    """
    data = signal['data'].transpose(2, 0, 1) # dm5储存的data为 frames x height x width，而velox储存的data为 height x width x frames
    width = data.shape[2] # width (X) 实际对应 [0]
    height = data.shape[1] # height (Y) 实际对应 [1]
    # 计算缩略图的尺寸
    width_0, height_0 = (384, 384)
    if width > height:
        height_0 = int(384/width*height) # height (Y) 实际对应 [1]
    elif width < height:
        width_0 = int(384/height*width) # width (X) 实际对应 [0]
    # 获取 frames
    frames = data.shape[0] # frames 实际对应 [2]
    if data.dtype == 'uint16':
        datatype = np.uint32(10) # 1 代表 signed int，10 代表 unsigned int
        pixeldepth = np.uint32(2) # 2 bytes 对应 16 bits
    elif data.dtype == 'float32':
        datatype = np.uint32(2) # 2 代表 float
        pixeldepth = np.uint32(4) # 4 bytes 对应 32 bits
    elif data.dtype == 'int16':
        datatype = np.uint32(1) # 1 代表 signed int，10 代表 unsigned int
        pixeldepth = np.uint32(2) # 2 bytes 对应 16 bits
    pixelsize = np.float32(parameters['pixelsize'])
    pixelunit = np.bytes_(parameters['pixelunit'].encode('gb2312')) # 中文系统不能用 utf-8
    keys_to_remove = {'data', 'metadata'}
    display_metadata = {k: v for k, v in signal.items() if k not in keys_to_remove}
    # 确定 image_display_info.attrs['HighLimit'] and ['LowLimit']
    low_limit = np.percentile(data[0,:,:], 0.1) # np.float64
    high_limit = np.percentile(data[0,:,:], 99.9) # np.float64
    low_limit = np.float32(min(signal['display_range']))
    high_limit = np.float32(max(signal['display_range']))
    # 构造 CLUT
    rgb = [('red', '<u2'), ('green', '<u2'), ('blue', '<u2')]
    base = np.arange(256, dtype=np.uint16)
    clut = np.zeros((256,), dtype=rgb)
    clut['red'] = int(257*signal['color']['red']) * base
    clut['green'] = int(257*signal['color']['green']) * base
    clut['blue'] = int(257*signal['color']['blue']) * base
    clutname = np.bytes_(b'Custom')
    if signal['color'] == {'blue': 1, 'green': 1, 'red': 1}:
        clutname = np.bytes_(b'Greyscale')
    # 图像名字
    img0_name = np.bytes_(f"Image of {Path(filename).stem}".encode('utf-8'))
    img1_name = Path(filename).stem.encode('utf-8')
    with h5py.File(filename, 'w') as f:
        # 添加根目录属性
        # f.attrs['ApplicationBounds'] = np.void((0, 0, 1545, 2418), dtype=[('top', '<i8'), ('left', '<i8'), ('bottom', '<i8'), ('right', '<i8')]) # '<i8' 表示 np.int64
        # f.attrs['BackgroundColor'] = np.void((62965, 62965, 64250), dtype=[('red', '<u2'), ('green', '<u2'), ('blue', '<u2')]) # '<u2' 表示 np.uint16
        # f.attrs['HasWindowPosition'] = np.uint8(1)
        f.attrs['InImageMode'] = np.uint8(1)
        # f.attrs['LayoutType'] = np.bytes_(b'Unknown')
        # f.attrs['NextDocumentObjectID'] = np.uint32(10)
        # f.attrs['WindowPosition'] = np.void((45, 11, 1534, 1500), dtype=[('top', '<i8'), ('left', '<i8'), ('bottom', '<i8'), ('right', '<i8')])

        # DocumentObjectList / [0] / ImageDisplayInfo
        doc_obj_list = f.create_group('DocumentObjectList')
        doc_obj_0 = doc_obj_list.create_group('[0]')
        
        # [0] 属性共 18 个
        doc_obj_0.attrs['AnnotationType'] = np.uint32(20)
        doc_obj_0.attrs['BackgroundColor'] = np.void((65535, 65535, 65535), dtype=[('red', '<u2'), ('green', '<u2'), ('blue', '<u2')])
        doc_obj_0.attrs['BackgroundMode'] = np.int16(2)
        doc_obj_0.attrs['FillMode'] = np.int16(1)
        doc_obj_0.attrs['ForegroundColor'] = np.void((0, 0, 0), dtype=[('red', '<u2'), ('green', '<u2'), ('blue', '<u2')])
        doc_obj_0.attrs['HasBackground'] = np.uint8(0)
        doc_obj_0.attrs['ImageDisplayType'] = np.int32(1) # 1 代表 2D raster ImageDisplay，3 对标 LinePlot
        doc_obj_0.attrs['ImageSource'] = np.uint64(0) # 用于确认实际的 ImageData 在哪，如果是 0，则代表我们需要去读取 ImageSourceList / [0] / ImageRef
        doc_obj_0.attrs['IsMoveable'] = np.uint8(1)
        doc_obj_0.attrs['IsResizable'] = np.uint8(1)
        doc_obj_0.attrs['IsSelectable'] = np.uint8(1)
        doc_obj_0.attrs['IsTransferrable'] = np.uint8(1)
        doc_obj_0.attrs['IsTranslatable'] = np.uint8(1)
        doc_obj_0.attrs['IsVisible'] = np.uint8(1)
        doc_obj_0.attrs['Rectangle'] = np.void((0.0, 0.0, np.float32(height), np.float32(width)), dtype=[('top', '<f4'), ('left', '<f4'), ('bottom', '<f4'), ('right', '<f4')]) # '<f4' 表示 np.float32
        doc_obj_0.attrs['RestrictionStyle'] = np.uint32(0)
        doc_obj_0.attrs['Transparency'] = np.float64(0.0)
        doc_obj_0.attrs['UniqueID'] = np.uint32(8)
        # [0] 属性结束
        image_display_info = doc_obj_0.create_group('ImageDisplayInfo')
        # ImageDisplayInfo 属性共 29 个
        image_display_info.attrs['BrightColor'] = np.void((65535, 65535, 65535), dtype=[('red', '<u2'), ('green', '<u2'), ('blue', '<u2')])
        image_display_info.attrs['Brightness'] = np.float32(0.5)
       #*# CLUT 名称 
        image_display_info.attrs['CLUTName'] = np.bytes_(b'Custom')
        image_display_info.attrs['CaptionOn'] = np.uint8(0)
        image_display_info.attrs['CaptionSize'] = np.uint16(14)
        image_display_info.attrs['ComplexMode'] = np.uint32(4)
        image_display_info.attrs['ComplexRange'] = np.float32(1000.0)
        image_display_info.attrs['Contrast'] = np.float32(0.5)
        image_display_info.attrs['ContrastAlgorithm'] = np.uint32(0)
        image_display_info.attrs['ContrastMode'] = np.uint32(1)
        image_display_info.attrs['DiffractionMode'] = np.uint8(0)
        image_display_info.attrs['DoAutoSurvey'] = np.uint8(1)
        image_display_info.attrs['EstimatedMax'] = low_limit # 不知道为什么 Max 对应最小的值 # 该数值无影响
        image_display_info.attrs['EstimatedMaxTrimPercentage'] = np.float32(0.001)
        image_display_info.attrs['EstimatedMin'] = high_limit # 不知道为什么 Min 对应最大的值 # 该数值无影响
        image_display_info.attrs['EstimatedMinTrimPercentage'] = np.float32(0.001)
        image_display_info.attrs['Gamma'] = np.float32(0.5)
        image_display_info.attrs['HiLimitContrastDeltaTriggerPercentage'] = np.float32(0.1)
        image_display_info.attrs['HighLimit'] = high_limit
        image_display_info.attrs['IsIgnoreGirdBar'] = np.uint8(0)
        image_display_info.attrs['IsInverted'] = np.uint8(0)
        image_display_info.attrs['LowLimit'] = low_limit
        image_display_info.attrs['LowLimitContrastDeltaTriggerPercentage'] = np.float32(0.1)
        image_display_info.attrs['MinimumContrast'] = np.float32(0.0)
        image_display_info.attrs['RangeAdjust'] = np.float32(1.0)
        image_display_info.attrs['SparseSurvey_GridSize'] = np.uint32(32)
        image_display_info.attrs['SparseSurvey_NumberPixels'] = np.uint32(64)
        image_display_info.attrs['SparseSurvey_UseNumberPixels'] = np.uint8(1)
        image_display_info.attrs['SurveyTechique'] = np.uint32(2)
        # ImageDisplayInfo 属性结束
       #*# CLUT
        image_display_info.create_dataset('CLUT', data=clut)
        # <HDF5 dataset "CLUT": shape (256,), type "|V6">
        # CLUT 结束
        image_display_info.create_group('MainSliceId')
        image_display_info['MainSliceId'].attrs['[0]'] = np.uint32(0)

        # Image Behavior 下的组
        image_behavior = f.create_group('Image Behavior')
        image_behavior.attrs['ImageDisplayBounds'] = np.void((0.0, 0.0, np.float32(height), np.float32(width)), dtype=[('top', '<f4'), ('left', '<f4'), ('bottom', '<f4'), ('right', '<f4')])
        image_behavior.attrs['IsZoomedToWindow'] = np.uint8(1)
        image_behavior.attrs['ViewDisplayID'] = np.uint32(8)
        image_behavior.attrs['WindowRect'] = np.void((0.0, 0.0, np.float32(height), np.float32(width)), dtype=[('top', '<f4'), ('left', '<f4'), ('bottom', '<f4'), ('right', '<f4')])
        
        image_behavior.create_group('UnscaledTransform')
        image_behavior['UnscaledTransform'].attrs['Offset'] = np.void((0.0, 0.0), dtype=[('x', '<f4'), ('y', '<f4')])
        image_behavior['UnscaledTransform'].attrs['Scale'] = np.void((1.0, 1.0), dtype=[('x', '<f4'), ('y', '<f4')])
        image_behavior.create_group('ZoomAndMoveTransform')
        image_behavior['ZoomAndMoveTransform'].attrs['Offset'] = np.void((0.0, 0.0), dtype=[('x', '<f4'), ('y', '<f4')]) # 当图像实际尺寸为HY255xWX264，Rectangle尺寸为HY689xWX713.3176，此时Offset为X0.34118652, Y0.0，Thumbnails的X,Y为713,689
        image_behavior['ZoomAndMoveTransform'].attrs['Scale'] = np.void((1.0, 1.0), dtype=[('x', '<f4'), ('y', '<f4')])

        # ImageList 组
        image_list = f.create_group('ImageList')

        # ImageList / [0]
        img0 = image_list.create_group('[0]')
        img0.attrs['Name'] = img0_name
        img0_image_data = img0.create_group('ImageData')
        # ImageData 属性
        img0_image_data.attrs['DataType'] = np.uint32(23) # DataType = 23 表示 RGBA，RGB 以及 Alpha 每个通道 8 bits，虽然 alpha 通道没有有用的信息
        img0_image_data.attrs['PixelDepth'] = np.uint32(4) # RGBA 每个像素需要 4 bytes
        # Calibrations（默认值，thumbnails 不需要 calibration）
        img0_calibrations = img0_image_data.create_group('Calibrations')
        img0_calibrations.attrs['DisplayCalibratedUnits'] = np.uint8(1)
        # Brightness（默认值，thumbnails 不需要 calibration）
        img0_calibrations.create_group('Brightness')
        img0_calibrations['Brightness'].attrs['Label'] = np.bytes_(b'')
        img0_calibrations['Brightness'].attrs['Origin'] = np.float32(0.0)
        img0_calibrations['Brightness'].attrs['Scale'] = np.float32(1.0)
        img0_calibrations['Brightness'].attrs['Units'] = np.bytes_(b'')
        # Dimension（默认值，thumbnails 不需要 calibration）
        img0_dimension = img0_calibrations.create_group('Dimension')
        img0_dimension.create_group('[0]')
        img0_dimension['[0]'].attrs['Label'] = np.bytes_(b'')
        img0_dimension['[0]'].attrs['Origin'] = np.float32(0.0)
        img0_dimension['[0]'].attrs['Scale'] = np.float32(1.0)
        img0_dimension['[0]'].attrs['Units'] = np.bytes_(b'')
        img0_dimension.create_group('[1]')
        img0_dimension['[1]'].attrs['Label'] = np.bytes_(b'')
        img0_dimension['[1]'].attrs['Origin'] = np.float32(0.0)
        img0_dimension['[1]'].attrs['Scale'] = np.float32(1.0)
        img0_dimension['[1]'].attrs['Units'] = np.bytes_(b'')
        # [0] image 的实际数据
        img0_image_data.create_dataset('Data', data=np.zeros(shape=(height_0, width_0, 3), dtype=np.uint8)) # [0] image 是一个 thumbnail，可能是为了在 Image Browser 中显示，格式为 RGB，长边尺寸为 384
        # [0] image 的实际数据的属性
        img0_image_data['Data'].attrs['CLASS'] = np.bytes_(b'IMAGE')
        img0_image_data['Data'].attrs['IMAGE_COLORMODEL'] = np.bytes_(b'RGB')
        img0_image_data['Data'].attrs['IMAGE_SUBCLASS'] = np.bytes_(b'IMAGE_TRUECOLOR')
        img0_image_data['Data'].attrs['INTERLACE_MODE'] = np.bytes_(b'INTERLACE_PIXEL')
        # [0] image 的大小尺寸
        img0_image_data.create_group('Dimensions')
        img0_image_data['Dimensions'].attrs['[0]'] = np.uint32(width_0) # width (X) 实际对应 [0]
        img0_image_data['Dimensions'].attrs['[1]'] = np.uint32(height_0) # height (Y) 实际对应 [1]
        # [0] ImageTags
        img0.create_group('ImageTags').create_group('GMS Version')
        img0['ImageTags']['GMS Version'].attrs['Created'] = np.bytes_(b'3.61.4682.0')
        # [0] UniqueID
        img0.create_group('UniqueID')
        # 目前尚不清楚 UniqueID 如何生成，以及是否与 ImageData 数据本身有关系，而且为什么有 4 个？根据官网的描述，4个数字一起组成一个 unique ID
        img0['UniqueID'].attrs['[0]'] = np.random.randint(100000000, np.iinfo(np.uint32).max, dtype=np.uint32)
        img0['UniqueID'].attrs['[1]'] = np.random.randint(100000000, np.iinfo(np.uint32).max, dtype=np.uint32)
        img0['UniqueID'].attrs['[2]'] = np.random.randint(100000000, np.iinfo(np.uint32).max, dtype=np.uint32)
        img0['UniqueID'].attrs['[3]'] = np.random.randint(100000000, np.iinfo(np.uint32).max, dtype=np.uint32)

        # ImageList / [1]
        img1 = image_list.create_group('[1]')
        img1.attrs['Name'] = img1_name
        img1_image_data = img1.create_group('ImageData')
       #*# ImageData 属性 #如果dtype是float32该怎么表示 
        img1_image_data.attrs['DataType'] = datatype
        img1_image_data.attrs['PixelDepth'] = pixeldepth
        # Calibrations
        img1_calibrations = img1_image_data.create_group('Calibrations')
        img1_calibrations.attrs['DisplayCalibratedUnits'] = np.uint8(1)
        # Brightness
        img1_calibrations.create_group('Brightness')
        img1_calibrations['Brightness'].attrs['Label'] = np.bytes_(b'')
        img1_calibrations['Brightness'].attrs['Origin'] = np.float32(0.0)
        img1_calibrations['Brightness'].attrs['Scale'] = np.float32(1.0)
        img1_calibrations['Brightness'].attrs['Units'] = np.bytes_(b'')
       #*# Dimension
        img1_dimension = img1_calibrations.create_group('Dimension')
        img1_dimension.create_group('[0]') # width (X) 实际对应 [0]
        img1_dimension['[0]'].attrs['Label'] = np.bytes_('X'.encode('utf-8'))
        img1_dimension['[0]'].attrs['Origin'] = np.float32(0.0)
        img1_dimension['[0]'].attrs['Scale'] = pixelsize
        img1_dimension['[0]'].attrs['Units'] = pixelunit
        img1_dimension.create_group('[1]') # height (Y) 实际对应 [1]
        img1_dimension['[1]'].attrs['Label'] = np.bytes_('Y'.encode('utf-8'))
        img1_dimension['[1]'].attrs['Origin'] = np.float32(0.0)
        img1_dimension['[1]'].attrs['Scale'] = pixelsize
        img1_dimension['[1]'].attrs['Units'] = pixelunit
        # series 文件中，[2] 对应时间
        img1_dimension.create_group('[2]') 
        img1_dimension['[2]'].attrs['Label'] = np.bytes_('Frames'.encode('utf-8'))
        img1_dimension['[2]'].attrs['Origin'] = np.float32(0.0)
        img1_dimension['[2]'].attrs['Scale'] = np.float32(1.0)
        img1_dimension['[2]'].attrs['Units'] = np.bytes_(''.encode('utf-8'))
       #*# [1] image 的实际数据
        img1_image_data.create_dataset('Data', data=data)
       #*# [1] image 的大小尺寸
        img1_image_data.create_group('Dimensions')
        img1_image_data['Dimensions'].attrs['[0]'] = np.uint32(width) # width (X) 实际对应 [0]
        img1_image_data['Dimensions'].attrs['[1]'] = np.uint32(height) # height (Y) 实际对应 [1]
        img1_image_data['Dimensions'].attrs['[2]'] = np.uint32(frames)
        # [1] ImageTags
        img1_tags = img1.create_group('ImageTags')
        gms_version = img1_tags.create_group('GMS Version')
        gms_version.attrs['Created'] = np.bytes_(b'3.61.4682.0')
        gms_version.attrs['Saved'] = np.bytes_(b'3.61.4682.0')
       #*# [1] Metadata
        metadata = img1_tags.create_group('Metadata')
        create_metadata(parameters, metadata)
        display_metadata_tags = img1_tags.create_group('Display Metadata')
        create_metadata(display_metadata, display_metadata_tags)
        original_metadata = img1_tags.create_group('Original Metadata')
        create_metadata(signal['metadata'], original_metadata)
        # [1] UniqueID
        img1.create_group('UniqueID')
        # 目前尚不清楚 UniqueID 如何生成，以及是否与 ImageData 数据本身有关系，而且为什么有 4 个？
        img1['UniqueID'].attrs['[0]'] = np.random.randint(100000000, np.iinfo(np.uint32).max, dtype=np.uint32)
        img1['UniqueID'].attrs['[1]'] = np.random.randint(100000000, np.iinfo(np.uint32).max, dtype=np.uint32)
        img1['UniqueID'].attrs['[2]'] = np.random.randint(100000000, np.iinfo(np.uint32).max, dtype=np.uint32)
        img1['UniqueID'].attrs['[3]'] = np.random.randint(100000000, np.iinfo(np.uint32).max, dtype=np.uint32)

        # ImageSourceList / [0]
        image_source_list = f.create_group('ImageSourceList')
        image_source_list.create_group('[0]').create_group('Id')
        # 适用于 Series 文件的 sum
        image_source_list['[0]'].attrs['ClassName'] = np.bytes_(b'ImageSource:Summed')
        image_source_list['[0]'].attrs['Do Sum'] = np.uint8(1)
        image_source_list['[0]'].attrs['ImageRef'] = np.uint32(1) # 这里显示 1，代表 ImageList / [1] 是我们需要的 ImageData
        image_source_list['[0]'].attrs['LayerEnd'] = np.uint32(0)
        image_source_list['[0]'].attrs['LayerStart'] = np.uint32(0)
        image_source_list['[0]'].attrs['Summed Dimension'] = np.uint32(2)
        # [0] Id 属性
        image_source_list['[0]']['Id'].attrs['[0]'] = np.uint32(0)

        # Page Behavior / PageTransform
        page_behavior = f.create_group('Page Behavior')
        # Page Behavior 属性共 8 个
        page_behavior.attrs['DrawMargins'] = np.uint8(1)
        page_behavior.attrs['DrawPaper'] = np.uint8(1)
        page_behavior.attrs['IsFixedInPageMode'] = np.uint8(0)
        page_behavior.attrs['IsZoomedToWindow'] = np.uint8(1)
        page_behavior.attrs['LayedOut'] = np.uint8(0)
        page_behavior.attrs['RestoreImageDisplayBounds'] = np.void((0.0, 0.0, 4096.0, 4096.0), dtype=[('top', '<f4'), ('left', '<f4'), ('bottom', '<f4'), ('right', '<f4')])
        page_behavior.attrs['RestoreImageDisplayID'] = np.uint32(8)
        page_behavior.attrs['TargetDisplayID'] = np.uint32(4294967295)
        # Page Behavior 属性结束
        page_behavior.create_group('PageTransform')
        page_behavior['PageTransform'].attrs['Offset'] = np.void((0.0, 0.0), dtype=[('x', '<f4'), ('y', '<f4')])
        page_behavior['PageTransform'].attrs['Scale'] = np.void((1.0, 1.0), dtype=[('x', '<f4'), ('y', '<f4')])

        # Thumbnails / [0]
        thumbnails = f.create_group('Thumbnails')
        thumbnails.create_group('[0]')
        # Thumbnails [0] 属性
        thumbnails['[0]'].attrs['ImageIndex'] = np.uint32(0)
        thumbnails['[0]'].attrs['SourceSize_Pixels'] = np.void((height, width), dtype=[('x', '<i8'), ('y', '<i8')]) # 当图像实际尺寸为HY255xWX264，Rectangle尺寸为HY689xWX713.3176，此时Offset为X0.34118652, Y0.0，Thumbnails的X,Y为713,689

def create_metadata(d, tags):
    if not isinstance(d, dict):
        return
    for key, value in d.items():
        if key == "CustomProperties" or key == "Operations" or key == "Features":
            continue
        if isinstance(value, dict):
            sub_tags = tags.create_group(key)
            create_metadata(value, sub_tags)
        elif isinstance(value, (list, tuple)):
            sub_tags = tags.create_group(key)
            for i, v in enumerate(value):
                try:
                    if isinstance(v, dict):
                        sub_sub_tags = sub_tags.create_group(f'[{i}]')
                        create_metadata(v, sub_sub_tags)
                    else:
                        sub_tags.attrs[f'[{i}]'] = v
                except ValueError:
                    print(f"group {key}/[{i}] already exists under {tags}")
                    continue
        else:
            try:
                if value == 0:
                    tags.attrs[key] = value
                elif not value:
                    tags.attrs[key] = b''
                else:
                    tags.attrs[key] = value
            except:
                print(f"{key}({type(key)}): {value}({type(value)})")

def save_image_as_png(
    image: np.ndarray,
    output_path: str,
    pixel_size: float = 1.0,
    pixel_unit: str = 'µm',
    title: str = None,
    cmap: str = 'gray',
    display_range: tuple = None,
    gamma: float = 1.0,
    display_index: int = 0,
    add_scalebar: bool = True,
    scalebar_length: float = None,
    scalebar_color: str = 'black',
    scalebar_thickness: int = None,
    scalebar_padding: float = 0.02,  # 图像高度/宽度的百分比
    scalebar_position: str = 'below',  # 'below' 或 'above'
    dpi: int = 300,
    transparent: bool = True,
    **kwargs
) -> bool:
    """
    修复版本：将2D数组保存为PNG，比例尺在图像外部，不改变图像宽度
    
    Args:
        image: 2D/3D numpy数组
        output_path: 输出PNG文件路径
        pixel_size: 像素尺寸
        pixel_unit: 物理尺寸单位
        title: 标题
        cmap: 颜色映射
        display_range: (vmin, vmax) 显示范围
        gamma: Gamma 校正值
        display_index: 3D图像的切片索引
        add_scalebar: 是否添加比例尺
        scalebar_length: 比例尺长度（物理单位）
        scalebar_color: 比例尺颜色
        scalebar_thickness: 比例尺厚度（像素）
        scalebar_padding: 比例尺与图像的间距（图像尺寸的百分比）
        scalebar_position: 比例尺位置 'below' 或 'above'
        dpi: 输出DPI
        transparent: 是否使用透明背景
        **kwargs: 其他参数
    
    Returns:
        bool: 保存是否成功
    """
    try:
        # 处理输出路径
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 提取2D图像
        if len(image.shape) == 3:
            image = image[:, :, display_index]
        elif len(image.shape) != 2:
            raise ValueError(f"图像必须是2D或3D数组，当前形状: {image.shape}")
        
        height, width = image.shape
        
        # 计算显示范围
        if display_range:
            vmin, vmax = display_range
        else:
            vmin, vmax = np.min(image), np.max(image)
        
        # 计算图像物理尺寸
        phys_width = width * pixel_size
        phys_height = height * pixel_size
        
        # 自动计算比例尺长度
        if add_scalebar and pixel_size != 1.0 and pixel_unit != 'pixel':
            if scalebar_length is None:
                # 选择最接近图像宽度30%的标准长度
                possible_lengths = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
                target_length = phys_width * 0.3
                scalebar_length = min(possible_lengths, key=lambda x: abs(x - target_length))
        
        # 计算比例尺在图像中的像素长度
        if add_scalebar:
            scalebar_px = int(scalebar_length / pixel_size)
            print(f"计算得到 scalebar 长度为 {scalebar_length} {pixel_unit} / {scalebar_px} pixels")
        else:
            scalebar_px = 0
        
        # 计算字体大小
        fontsize_pt = height // 50
        print(f"计算得到 font size 为 {fontsize_pt} pixels")
        
        # 计算比例尺厚度
        if scalebar_thickness is None:
            # 基于图像高度和scalebar像素长度计算
            scalebar_thickness = max(scalebar_px*0.15, int(height * 0.05))  # 图像高度的5%
            print(f"计算得到 scalebar 厚度为 {scalebar_thickness} pixels")
        
        # 计算比例尺区域的额外高度
        scalebar_area_height = 0
        if add_scalebar:
            # 比例尺区域高度 = 比例尺厚度 + 字体高度 + 间距
            # 字体高度大约是字体大小的1.2倍（点数转换为像素：1点 = 1/72英寸）
            font_height_px = (fontsize_pt / 72) * dpi
            scalebar_area_height = int(scalebar_thickness + font_height_px * 1.5 + height * scalebar_padding)
        
        # 创建图形，宽度不变，高度增加比例尺区域
        fig_width_inches = width / dpi
        fig_height_inches = (height + scalebar_area_height) / dpi
        
        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=dpi)
        
        # 创建坐标轴，图像占据上方区域，比例尺在下方（或上方）
        if scalebar_position == 'below':
            # 图像在上，比例尺在下
            image_top = 1.0
            image_bottom = scalebar_area_height / (height + scalebar_area_height)
            scalebar_top = image_bottom
            scalebar_bottom = 0
        else:  # 'above'
            # 比例尺在上，图像在下
            scalebar_top = 1.0
            scalebar_bottom = 1.0 - (scalebar_area_height / (height + scalebar_area_height))
            image_top = scalebar_bottom
            image_bottom = 0
        
        # 图像坐标轴
        ax_image = fig.add_axes([0, image_bottom, 1, image_top - image_bottom])
        
        # 使用 PowerNorm 进行 Gamma 校正
        if gamma != 1.0:
            norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
        else:
            norm = None
        
        # 显示图像
        im = ax_image.imshow(
            image,
            cmap=cmap,
            norm=norm,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
            aspect='equal',
            interpolation='nearest'
        )
        
        # 移除图像坐标轴
        ax_image.set_axis_off()
        
        # 添加标题（如果有）
        if title:
            ax_image.set_title(title, fontsize=fontsize_pt, pad=10, color='black')
        
        # 添加比例尺
        if add_scalebar and pixel_size != 1.0 and pixel_unit != 'pixel':
            # 比例尺坐标轴
            ax_scalebar = fig.add_axes([0, scalebar_bottom, 1, scalebar_top - scalebar_bottom])
            
            # 设置白色背景
            # ax_scalebar.set_facecolor('white')
            
            # 添加比例尺
            add_scalebar_to_axis(
                ax=ax_scalebar,
                pixel_size=pixel_size,
                pixel_unit=pixel_unit,
                image_width=width,
                scalebar_length=scalebar_length,
                scalebar_px=scalebar_px,
                scalebar_color=scalebar_color,
                scalebar_thickness=scalebar_thickness,
                fontsize=fontsize_pt,
                position=scalebar_position,
                dpi=dpi
            )
            
            # 移除比例尺坐标轴的边框和刻度
            ax_scalebar.set_axis_off()
        
        # 保存为PNG
        plt.savefig(
            output_path,
            dpi=dpi,
            transparent=transparent,
            bbox_inches='tight',
            pad_inches=0,
            format='png'
        )
        
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"保存PNG时出错: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False

def resize_image_to_min_512(image: np.ndarray, interpolation: int = Image.LANCZOS) -> tuple:
    """
    如果图像任意一边小于512像素，则等比例放大到短边=512像素
    
    Args:
        image: 输入图像 (numpy数组, 值范围0-255或0-1)
        interpolation: PIL插值方法
        
    Returns:
        tuple: (缩放后的图像, 缩放比例)
    """
    height, width = image.shape[:2]
    
    # 检查是否需要放大
    if height < 512 or width < 512:
        # 计算缩放比例
        scale = 512 / min(height, width)
        
        # 计算新的尺寸
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # 确保图像值在0-255范围内（PIL需要）
        if image.dtype != np.uint8:
            # 如果图像是浮点数，假设范围是0-1，转换为0-255
            if image.max() <= 1.0:
                image_rescaled = (image * 255).astype(np.uint8)
            else:
                image_rescaled = image.astype(np.uint8)
        else:
            image_rescaled = image
        
        # 使用PIL进行高质量的图像缩放
        pil_image = Image.fromarray(image_rescaled)
        resized_pil = pil_image.resize((new_width, new_height), interpolation)
        
        # 转换回numpy数组
        resized_image = np.array(resized_pil)
        
        print(f"图像从 {height}x{width} 放大到 {new_height}x{new_width}，缩放比例: {scale:.2f}")
        return resized_image, scale
    
    return image, 1.0

def save_color_mix_image(
    image: np.ndarray,
    output_path: str = None,
    line_info: dict = None,
    pixel_size: float = 1.0,
    pixel_unit: str = 'µm',
    add_scalebar: bool = True,
    scalebar_length: float = None,
    scalebar_color: str = 'black',
    scalebar_thickness: int = None,
    scalebar_padding: float = 0.02,  # 图像高度/宽度的百分比
    scalebar_position: str = 'below',  # 'below' 或 'above'
    dpi: int = 300,
    transparent: bool = True,
    min_image_size: int = 512,  # 新增参数：最小图像尺寸
    resize_interpolation: str = 'lanczos',  # 新增参数：插值方法
    auto_resize: bool = True,  # 新增参数：是否自动调整大小
    **kwargs
):
    """
    优化版本：自动调整图像尺寸并保存为PNG，比例尺在图像外部
    
    Args:
        image: 3D numpy数组 (RGB图像)
        output_path: 输出PNG文件路径
        pixel_size: 像素尺寸
        pixel_unit: 物理尺寸单位
        add_scalebar: 是否添加比例尺
        scalebar_length: 比例尺长度（物理单位）
        scalebar_color: 比例尺颜色
        scalebar_thickness: 比例尺厚度（像素）
        scalebar_padding: 比例尺与图像的间距（图像尺寸的百分比）
        scalebar_position: 比例尺位置 'below' 或 'above'
        dpi: 输出DPI
        transparent: 是否使用透明背景
        min_image_size: 图像最小尺寸（短边），默认512像素
        resize_interpolation: 图像放大时的插值方法
        auto_resize: 是否自动调整图像大小
        **kwargs: 其他参数
    
    Returns:
        bool or Figure: 保存成功返回True，否则返回False；如果没有输出路径则返回Figure对象
    """
    try:
        # 保存原始参数（用于后续恢复）
        original_image = image.copy()
        original_pixel_size = pixel_size
        original_line_info = line_info.copy() if line_info else None
        
        # 1. 如果需要，检查并调整图像尺寸
        if auto_resize:
            # 映射插值方法字符串到PIL常量
            interpolation_map = {
                'nearest': Image.NEAREST,
                'bilinear': Image.BILINEAR,
                'bicubic': Image.BICUBIC,
                'lanczos': Image.LANCZOS,
                'box': Image.BOX,
                'hamming': Image.HAMMING
            }
            
            interpolation = interpolation_map.get(
                resize_interpolation.lower() if isinstance(resize_interpolation, str) else 'lanczos',
                Image.LANCZOS
            )
            
            image, scale_factor = resize_image_to_min_512(image, interpolation)
            
            # 根据缩放因子调整相关参数
            if scale_factor != 1.0:
                # 调整像素尺寸（因为图像放大了，每个像素代表的物理尺寸变小）
                pixel_size = original_pixel_size / scale_factor
                
                # 调整线条信息（如果存在）
                if line_info:
                    # 缩放线条坐标
                    line_info['start'] = (
                        int(line_info['start'][0] * scale_factor),
                        int(line_info['start'][1] * scale_factor)
                    )
                    line_info['end'] = (
                        int(line_info['end'][0] * scale_factor),
                        int(line_info['end'][1] * scale_factor)
                    )
                    # 缩放线宽
                    line_info['line_width'] = int(line_info['line_width'] * scale_factor)
        else:
            scale_factor = 1.0
        
        # 2. 处理输出路径
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 3. 验证图像形状
        if len(image.shape) != 3:
            raise ValueError(f"图像必须是3D数组，当前数组形状: {image.shape}")
        if image.shape[-1] != 3:
            raise ValueError(f"图像必须是RGB，当前数组形状: {image.shape}")
        
        height, width = image.shape[:2]
        
        # 4. 计算图像物理尺寸
        phys_width = width * pixel_size
        phys_height = height * pixel_size
        
        # 5. 自动计算比例尺长度
        if add_scalebar and pixel_size != 1.0 and pixel_unit != 'pixel':
            if scalebar_length is None:
                # 选择最接近图像宽度30%的标准长度
                possible_lengths = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
                target_length = phys_width * 0.3
                scalebar_length = min(possible_lengths, key=lambda x: abs(x - target_length))
        
        # 6. 计算比例尺在图像中的像素长度
        if add_scalebar and scalebar_length:
            scalebar_px = int(scalebar_length / pixel_size)
            print(f"计算得到 scalebar 长度为 {scalebar_length} {pixel_unit} / {scalebar_px} pixels")
        else:
            scalebar_px = 0
        
        # 7. 计算字体大小（基于图像高度）
        fontsize_pt = max(height // 50, 8)  # 设置最小字体大小
        print(f"计算得到 font size 为 {fontsize_pt} pixels")
        
        # 8. 计算比例尺厚度
        if scalebar_thickness is None and add_scalebar:
            # 基于图像高度和比例尺像素长度计算
            base_thickness = int(height * 0.02)  # 图像高度的2%
            if scalebar_px > 0:
                scalebar_thickness = max(int(scalebar_px * 0.1), base_thickness)
            else:
                scalebar_thickness = base_thickness
            print(f"计算得到 scalebar 厚度为 {scalebar_thickness} pixels")
        
        # 9. 计算比例尺区域的额外高度
        scalebar_area_height = 0
        if add_scalebar:
            # 比例尺区域高度 = 比例尺厚度 + 字体高度 + 间距
            font_height_px = (fontsize_pt / 72) * dpi
            scalebar_area_height = int(scalebar_thickness + font_height_px * 1.5 + height * scalebar_padding)
        
        # 10. 创建图形
        fig_width_inches = width / dpi
        fig_height_inches = (height + scalebar_area_height) / dpi
        
        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=dpi)
        
        # 11. 创建坐标轴布局
        if scalebar_position == 'below':
            # 图像在上，比例尺在下
            image_top = 1.0
            image_bottom = scalebar_area_height / (height + scalebar_area_height)
            scalebar_top = image_bottom
            scalebar_bottom = 0
        else:  # 'above'
            # 比例尺在上，图像在下
            scalebar_top = 1.0
            scalebar_bottom = 1.0 - (scalebar_area_height / (height + scalebar_area_height))
            image_top = scalebar_bottom
            image_bottom = 0
        
        # 12. 图像坐标轴
        ax_image = fig.add_axes([0, image_bottom, 1, image_top - image_bottom])
        
        # 显示图像（确保图像数据格式正确）
        if image.dtype != np.uint8 and image.max() <= 1.0:
            # 如果是0-1范围的浮点数，转换为0-255
            display_image = (image * 255).astype(np.uint8)
        else:
            display_image = image.astype(np.uint8)
        
        im = ax_image.imshow(
            display_image,
            aspect='equal',
            interpolation='nearest'
        )
        
        # 13. 提取并绘制线条信息（像素坐标）
        if line_info:
            start_px = line_info['start']  # (x, y) 像素坐标
            end_px = line_info['end']      # (x, y) 像素坐标
            line_width_px = line_info['line_width']
            # print(f"Line Annotation: Start {start_px} -> End {end_px}")
            # print(f"Line Annotation: Width {line_width_px}")
        
            # 转换为物理坐标
            # start_phys = (start_px[0] * pixel_size, start_px[1] * pixel_size)
            start_phys = start_px
            # end_phys = (end_px[0] * pixel_size, end_px[1] * pixel_size)
            end_phys = end_px
            print(f"Line Annotation: Start {start_phys} -> End {end_phys}")
            print(f"x轴限制: {ax_image.get_xlim()}")
            print(f"y轴限制: {ax_image.get_ylim()}")
        
            # 计算线宽的一半（用于绘制方框）
            # half_width = line_width_px * pixel_size / 2
            half_width = line_width_px / 2
            print(f"Line Annotation: Width {half_width}")
        
            # 计算线条方向向量和垂直向量
            dx = end_phys[0] - start_phys[0]
            dy = end_phys[1] - start_phys[1]
            line_length = np.sqrt(dx**2 + dy**2)
        
            # 归一化方向向量
            if line_length > 0:
                ux = dx / line_length
                uy = dy / line_length
            else:
                ux, uy = 1, 0
            
            # 垂直向量（逆时针旋转90度）
            vx = -uy
            vy = ux
            
            # 计算矩形（方框）的四个角点
            p1 = (start_phys[0] + vx * half_width, start_phys[1] + vy * half_width)
            p2 = (start_phys[0] - vx * half_width, start_phys[1] - vy * half_width)
            p3 = (end_phys[0] - vx * half_width, end_phys[1] - vy * half_width)
            p4 = (end_phys[0] + vx * half_width, end_phys[1] + vy * half_width)
            
            # 绘制矩形（方框表示线宽）
            rect_polygon = Polygon([p1, p2, p3, p4], closed=True, 
                                   fill=False, edgecolor='darkred', linewidth=0.7)
            ax_image.add_patch(rect_polygon)

            # 绘制箭头
            ax_image.annotate('', 
                             xy=end_phys, 
                             xytext=start_phys, 
                             arrowprops=dict(arrowstyle='->', 
                                           color='yellow', 
                                           linewidth=0.7, 
                                           shrinkA=0, 
                                           shrinkB=0), 
                             zorder=5)
        
        # 移除图像坐标轴
        ax_image.set_axis_off()
        
        # 14. 添加比例尺
        if add_scalebar and pixel_size != 1.0 and pixel_unit != 'pixel' and scalebar_length:
            # 比例尺坐标轴
            ax_scalebar = fig.add_axes([0, scalebar_bottom, 1, scalebar_top - scalebar_bottom])
            
            # 添加比例尺
            add_scalebar_to_axis(
                ax=ax_scalebar,
                pixel_size=pixel_size,
                pixel_unit=pixel_unit,
                image_width=width,
                scalebar_length=scalebar_length,
                scalebar_px=scalebar_px,
                scalebar_color=scalebar_color,
                scalebar_thickness=scalebar_thickness,
                fontsize=fontsize_pt,
                position=scalebar_position,
                dpi=dpi
            )
            
            # 移除比例尺坐标轴的边框和刻度
            ax_scalebar.set_axis_off()
        
        # 15. 保存图像
        if output_path:
            plt.savefig(
                output_path,
                dpi=dpi,
                transparent=transparent,
                bbox_inches='tight',
                pad_inches=0,
                format='png'
            )
            plt.close(fig)
            print(f"图像已保存到: {output_path}")
            return True
        else:
            return fig

    except Exception as e:
        print(f"保存PNG时出错: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False

def add_scalebar_to_axis(
    ax,
    pixel_size: float,
    pixel_unit: str,
    image_width: int,
    scalebar_length: float,
    scalebar_px: int,
    scalebar_color: str = 'black',
    scalebar_thickness: int = 5,
    fontsize: float = 12,
    position: str = 'below',
    dpi: int = 300,
    bg_color: str = 'white'
):
    """
    在坐标轴上添加比例尺，带白色背景
    
    Args:
        ax: matplotlib坐标轴对象
        pixel_size: 像素尺寸
        pixel_unit: 物理尺寸单位
        image_width: 图像宽度（像素）
        scalebar_length: 比例尺长度（物理单位）
        scalebar_px: 比例尺像素长度
        scalebar_color: 比例尺颜色
        scalebar_thickness: 比例尺厚度（像素）
        fontsize: 字体大小（点）
        position: 比例尺位置 'below' 或 'above'
        dpi: DPI
        bg_color: 背景颜色
    """
    # 设置坐标轴范围（像素坐标）
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, 1)
    
    # 设置白色背景
    # ax.set_facecolor(bg_color)
    
    # 计算比例尺位置（水平居中）
    x_center = image_width / 2
    x_start = x_center - scalebar_px / 2
    x_end = x_center + scalebar_px / 2
    
    # 根据位置设置y坐标
    if position == 'below':
        # 比例尺在下方，靠近底部显示
        y_pos = 0.3
        text_y = y_pos + 0.1  # 文字在比例尺上方
        text_va = 'bottom'  # 文字底部对齐
    else:  # 'above'
        # 比例尺在上方，靠近底部显示
        y_pos = 0.3
        text_y = y_pos + 0.2  # 文字在比例尺下方
        text_va = 'top'  # 文字顶部对齐
    
    # 计算比例尺厚度在坐标轴中的高度
    # 坐标轴高度为1，我们需要将像素厚度转换为坐标轴单位
    scalebar_height = scalebar_thickness / (ax.bbox.height * dpi / 72)  # 转换为坐标轴单位
    print(f"实际输出的 scalebar 厚度为 {scalebar_thickness} pixels\n实际输出的 scalebar 长度为 {scalebar_px} pixels")
    # 添加比例尺背景（白色）
    background_rect = Rectangle(
        (0, 0),  # 左下角坐标
        image_width,   # 宽度（像素）
        1,       # 高度（坐标轴单位，0-1）
        linewidth=0,
        edgecolor='white',
        facecolor='white',
        transform=ax.transData,
        zorder=-100  # 确保在最底层
    )
    ax.add_patch(background_rect)

    # 添加比例尺矩形（黑色）
    scalebar_rect = Rectangle(
        (x_start, y_pos - scalebar_height/2),
        scalebar_px,
        scalebar_height,
        linewidth=0,
        edgecolor='black',
        facecolor='black',
        transform=ax.transData,
        zorder=10
    )
    ax.add_patch(scalebar_rect)
    
    # 获取单位符号
    unit_symbol = get_unit_symbol(pixel_unit)
    
    # 格式化比例尺文本
    if scalebar_length < 1:
        # 小数值，保留适当小数位
        if scalebar_length < 0.01:
            scalebar_text = f"{scalebar_length:.3f} {unit_symbol}"
        elif scalebar_length < 0.1:
            scalebar_text = f"{scalebar_length:.2f} {unit_symbol}"
        else:
            scalebar_text = f"{scalebar_length:.1f} {unit_symbol}"
    else:
        # 整数值，显示整数
        if scalebar_length == int(scalebar_length):
            scalebar_text = f"{int(scalebar_length)} {unit_symbol}"
        else:
            scalebar_text = f"{scalebar_length:.1f} {unit_symbol}"
    print(f"实际输出的 font size 为 {fontsize}")
    # 添加比例尺文字（黑色）
    ax.text(
        x_center, text_y,
        scalebar_text,
        fontsize=fontsize,
        color='black',
        verticalalignment=text_va,
        horizontalalignment='center',
        transform=ax.transData,
        zorder=20,
        fontweight='bold'
    )

def get_unit_symbol(unit: str) -> str:
    """
    获取单位符号
    """
    unit_symbols = {
        'm': 'm',
        'meter': 'm',
        'meters': 'm',
        'cm': 'cm',
        'centimeter': 'cm',
        'centimeters': 'cm',
        'mm': 'mm',
        'millimeter': 'mm',
        'millimeters': 'mm',
        'µm': 'µm',
        'um': 'µm',
        'micrometer': 'µm',
        'micrometers': 'µm',
        'micron': 'µm',
        'microns': 'µm',
        'nm': 'nm',
        'nanometer': 'nm',
        'nanometers': 'nm',
        'Å': 'Å',
        'angstrom': 'Å',
        'angstroms': 'Å',
        'pixel': 'px',
        'pixels': 'px',
        'px': 'px',
    }
    
    if unit is None:
        return 'px'
    
    unit_str = str(unit).strip().lower()
    
    if unit_str in unit_symbols:
        return unit_symbols[unit_str]
    
    for key, value in unit_symbols.items():
        if key and unit_str.endswith(key.lower()):
            return value
    
    return unit

# ============================================================================
# Velox 文件分析器主类
# ============================================================================

class VeloxFileAnalyzer:
    """Velox EMD 文件分析器，支持多种数据类型的解析和提取。"""
    
    def __init__(self, file_path):
        """
        初始化分析器，加载文件并提取特征。
        
        参数:
            file_path: Velox EMD 文件路径
        """
        self.file_path = file_path
        try:
            self.f = h5py.File(file_path, 'r')
        except OSError:
            print("[ERROR] 文件已经被其他软件打开")
        if not hasattr(self, 'f'):
            raise ValueError("[ERROR] 文件已经被其他软件打开")
        self.features = bytes_to_json(self.f['Features']['Features'])['features']
        
        # 初始化各特征路径
        self._init_feature_paths()
        
        # 获取实验日志
        self.get_experiment_log()
        
        # 根据检测到的特征类型执行相应的数据提取
        self._extract_data_based_on_features()
    
    def _init_feature_paths(self):
        """初始化所有可能存在的特征路径。"""
        feature_types = {
            'ColorMixProfileFeature': 'color_mix_profile_feature_path',
            'IntegratedSpectraFeature': 'integrated_spectra_feature_path',
            'SIFeature': 'si_feature_path',
            'CameraFeature': 'camera_feature_path',
            'DcfiFeature': 'dcfi_feature_path',
            'STEMFeature': 'stem_feature_path',
            'DPCFeature': 'dpc_feature_path',
            'CropFeature': 'crop_feature_path',
            'ImageFilteringFeature': 'image_filter_feature_path'
        }
        
        for feature in self.features:
            for feature_type, attr_name in feature_types.items():
                if feature_type in feature:
                    setattr(self, attr_name, feature[feature_type])
                    break
    
    def _extract_data_based_on_features(self):
        """根据检测到的特征类型执行相应的数据提取方法。"""
        feature_handlers = [
            ('si_feature_path', self.get_element_maps_and_basic_settings),
            ('integrated_spectra_feature_path', self.extract_integrated_spectra),
            ('color_mix_profile_feature_path', self._handle_color_mix_and_line_profile),
            ('camera_feature_path', self.get_tem_image_and_settings),
            ('dcfi_feature_path', self.get_dcfi_image_and_settings),
            ('stem_feature_path', self.get_stem_image_and_settings),
            ('dpc_feature_path', self.get_dpc_images_and_settings),
            ('crop_feature_path', self.get_crop_image),
            ('image_filter_feature_path', self.get_filtered_image)
        ]
        
        for feature_attr, handler in feature_handlers:
            if hasattr(self, feature_attr):
                handler()
    
    def _handle_color_mix_and_line_profile(self):
        """处理颜色混合和线剖面数据。"""
        self.get_line_profile()
        self.get_color_mix_image()
    
    def get_path(self, path):
        """
        获取指定路径对应的数据内容。
        
        参数:
            path: HDF5 路径
        
        返回:
            路径对应的数据内容
        """
        if not path:
            return ""
        
        result = self.f[path]
        
        # 如果是特定格式的数据集，转换为 JSON
        if (isinstance(result, (h5py.Dataset, np.ndarray)) and
            result.dtype == object and result.shape == (1,)):
            result = bytes_to_json(result)
        
        return result
    
    # ========================================================================
    # STEM 图像处理
    # ========================================================================
    
    def get_stem_image_and_settings(self):
        """提取 STEM 图像数据和基本参数。"""
        if not hasattr(self, 'stem_feature_path'):
            print('[ERROR] 不是 STEM 图像')
            return self
        stem_feature = self.get_path(self.stem_feature_path)
        image_datas = {} # 储存数据
        image_metadatas = {} # 储存原始 metadata
        parameters = {} # 储存提取的 metadata
        for index, operation_path in enumerate(stem_feature['stemInputOperations']):
            stem_input_operation = self.get_path(operation_path)
            image_display = self.get_path(stem_feature['imageDisplays'][index])
            image_label = image_display['display']['label']
            detector_name = stem_input_operation['detector']
            stem_data = self.get_path(stem_input_operation['dataPath'])
            # 根据帧数确定图像类型
            if stem_data['FrameLookupTable'].shape[0] == 1:
                if index == 0:
                    print('[INFO] STEM 单张图像')
                image_data = stem_data['Data'][:, :, 0]  # 获取 2D 图像数据
                image_metadata = decode_metadata(stem_data['Metadata'])
            else:
                if index == 0:
                    print('[INFO] STEM 系列图像')
                data_path = stem_input_operation['dataPath'] + '/Data'
                image_data = optimized_read_with_progress(self.file_path, data_path)
                image_metadata = decode_metadata(stem_data['Metadata'])
            image_datas[image_label] = image_data
            image_metadatas[image_label] = image_metadata
            # 提取显示参数
            parameters[image_label] = {
                'detector_name': detector_name,
                'image_name': f"{image_label} of {Path(self.file_path).stem}",
                'pixelsize': self._get_pixel_size(image_metadata)[0],
                'pixelunit': self._get_pixel_size(image_metadata)[1],
                'display_index': int(image_display['seriesIndex']),
                'display_range': [
                    float(item) for item in image_display['displayLevelsRange'].values()
                ],
                'gamma': float(image_display['gamma'])
            }
        self.stem_data = image_datas
        self.stem_metadata = image_metadatas
        self._update_parameters(parameters)
        return self

    # ========================================================================
    # DPC 图像处理
    # ========================================================================
    
    def get_dpc_images_and_settings(self):
        """提取 DPC 图像数据和设置参数。"""
        if not hasattr(self, 'dpc_feature_path'):
            print('[ERROR] 不是 DPC 图像')
            return self
        dpc_feature = self.get_path(self.dpc_feature_path)
        # 提取所有图像数据
        image_displays = [
            self.get_path(item) for item in dpc_feature['imageDisplays']
        ]
        image_datas = {} # 储存数据
        image_metadatas = {} # 储存原始 metadata
        parameters = {} # 储存提取的 metadata
        for display in image_displays:
            label = display['display']['label']
            data_path = display['dataPath']
            data_obj = self.get_path(data_path)
            # image_datas[label] = data_obj['Data'][:, :, int(display['seriesIndex'])]
            image_datas[label] = data_obj['Data'][:, :, :]
            image_metadatas[label] = decode_metadata(data_obj['Metadata'])
            parameters[label] = {
                'pixelsize': self._get_pixel_size(image_metadatas[label])[0],
                'pixelunit': self._get_pixel_size(image_metadatas[label])[1],
                'image_name': f"{label} of {Path(self.file_path).stem}",
                'data_path': data_path,
                'display_index': int(display['seriesIndex']),
                'display_range': [
                    float(item) for item in display['displayLevelsRange'].values()
                ],
                'gamma': float(display['gamma'])
            }
        self.dpc_data = image_datas
        self.dpc_metadata = image_metadatas
        self._update_parameters(parameters)
        return self
    
    # ========================================================================
    # TEM 图像处理
    # ========================================================================
    
    def get_tem_image_and_settings(self):
        """提取 TEM 图像数据和基本参数。"""
        if not hasattr(self, 'camera_feature_path'):
            print('[ERROR] 不是 TEM 图像')
            return self
        camera_feature = self.get_path(self.camera_feature_path)
        camera_input_operation = self.get_path(camera_feature['cameraInputOperation'])
        camera_name = camera_input_operation['cameraName']
        camera_input_data = self.get_path(camera_input_operation['dataPath'])
        # 根据帧数确定图像类型
        if camera_input_data['FrameLookupTable'].shape[0] == 1:
            print('[INFO] TEM 单张图像')
            image_data = camera_input_data['Data'][:, :, 0]
            image_metadata = decode_metadata(camera_input_data['Metadata'])
        else:
            print('[INFO] TEM 系列图像')
            data_path = camera_input_operation['dataPath'] + '/Data'
            image_data = optimized_read_with_progress(self.file_path, data_path)
            image_metadata = decode_metadata(camera_input_data['Metadata'])
        self.tem_data = image_data
        self.tem_metadata = image_metadata
        # 提取显示参数
        parameters = {}
        image_display = self.get_path(camera_feature['imageDisplay'])
        parameters[camera_name] = {
            'camera_name': camera_name,
            'image_name': Path(self.file_path).stem,
            'pixelsize': self._get_pixel_size(image_metadata)[0],
            'pixelunit': self._get_pixel_size(image_metadata)[1],
            'display_index': int(image_display['seriesIndex']),
            'display_range': [
                float(item) for item in image_display['displayLevelsRange'].values()
            ],
            'gamma': float(image_display['gamma'])
        }
        self._update_parameters(parameters)
        return self
    
    # ========================================================================
    # DCFI 图像处理
    # ========================================================================
    
    def get_dcfi_image_and_settings(self):
        """提取 DCFI 图像和参数设置。"""
        if not hasattr(self, 'dcfi_feature_path'):
            print('[ERROR] 没有 DCFI 图像')
            return self
        dcfi_feature = self.get_path(self.dcfi_feature_path)
        dcfi_display = self.get_path(dcfi_feature['imageDisplay'])
        dcfi_data_path = dcfi_display['dataPath']
        # 读取 DCFI 数据
        dcfi_data = optimized_read_with_progress(
            self.file_path,
            dcfi_data_path + '/Data'
        )
        self.dcfi_data = dcfi_data
        image_metadata = decode_metadata(self.get_path(dcfi_data_path+'/Metadata'))
        # 提取显示参数
        parameters = {}
        parameters['DCFI'] = {
            'image_name': f"{unquote(dcfi_display['display']['label'])} of {Path(self.file_path).stem}",
            'pixelsize': self._get_pixel_size(image_metadata)[0],
            'pixelunit': self._get_pixel_size(image_metadata)[1],
            'display_index': int(dcfi_display['seriesIndex']),
            'display_range': [
                float(item) for item in dcfi_display['displayLevelsRange'].values()
            ],
            'gamma': float(dcfi_display['gamma'])
        }
        self._update_parameters(parameters)
        return self
    
    # ========================================================================
    # 实验日志
    # ========================================================================
    
    def get_experiment_log(self):
        """提取实验日志。"""
        result = bytes_to_json(self.f['Experiment']).get('log')
        text = bytes_to_json(self.f[result])['text']
        experiment_log = unquote(text)
        self.experiment_log = experiment_log
        return self
    
    # ========================================================================
    # 元素分布图处理
    # ========================================================================
    
    def get_element_maps_and_basic_settings(self):
        """提取所有元素分布图和 HAADF 图像，以及基本的扫描参数。"""
        si_feature = self.get_path(self.si_feature_path)
        
        # 提取 EDS 谱图数据
        self._extract_eds_spectra(si_feature)
        
        # 提取 STEM 图像和元素分布图
        self._extract_mapping_data(si_feature)
        
        # 获取元数据和其他参数
        self._extract_si_parameters()
        
        # 获取 color mix image
        self.get_color_mix_image()
        
        return self
    
    def _extract_eds_spectra(self, si_feature):
        """提取 EDS 谱图数据。"""
        eds_detector = si_feature['eds']['detectors'][0]
        eds_detector_segments = {}
        
        for item in eds_detector['segments']:
            if item['summed']:
                detector_key = f"{eds_detector['physicalDetector']}{item['index']}"
                segment_path = self.get_path(item['renderedSpectrum'])['dataPath']
                eds_detector_segments[detector_key] = self.get_path(segment_path)
        
        # 确定通道数
        if eds_detector_segments:
            first_segment = next(iter(eds_detector_segments.values()))
            channels = first_segment['Data'].shape[0]
        else:
            channels = 0
        
        # 提取谱图数据
        spectra_data = {}
        data_sum = np.zeros(channels, dtype=float)
        
        for key, value in eds_detector_segments.items():
            spectrum = value['Data'][:, 0]
            metadata = decode_metadata(value['Metadata'])
            data_sum += spectrum
            
            spectra_data[key] = {
                'spectrum': spectrum,
                'metadata': metadata,
            }
        
        spectra_data['total'] = data_sum
        self.spectra_data = spectra_data
    
    def _extract_mapping_data(self, si_feature):
        """提取映射数据。"""
        multi_image_display = self.get_path(si_feature['multiImageDisplay'])
        display_group_items = [
            self.get_path(item) for item in multi_image_display['displayGroupItems']
        ]
        
        mapping_data = {}
        
        for item in display_group_items:
            display = self.get_path(item['display'])
            data_obj = self.get_path(display['data'])
            data_path = data_obj['dataPath']
            
            data_item = self.get_path(data_path)
            
            mapping_data[display['id']] = {
                'data': data_item['Data'][()],
                'metadata': decode_metadata(data_item['Metadata']),
                'frame_index': int(data_obj['frameIndex']),
                'color': self.get_path(display['settings'])['color'],
                'display_range': [
                    float(num) for num in self.get_path(display['settings'])['displayLevelsRange'].values()
                ],
                'gamma': float(self.get_path(display['settings'])['gamma']),
                'blend_factor': item['blendFactor'],
                'blend_mode': item['blendMode'],
            }
            
            # 如果是 STEM 图像，提取额外的参数
            if item['groupType'] == 'Stem':
                metadata = decode_metadata(data_item['Metadata'])
                data = data_item['Data'][()]
                
                if not hasattr(self, 'parameters'):
                    self.parameters = {}
                
                if 'frames' not in self.parameters:
                    self.parameters['frames'] = int(
                        metadata['CustomProperties']['Velox.Series.FrameNumber']['value']
                    )
                
                if 'image_shape' not in self.parameters:
                    self.parameters['image_shape'] = data.shape[:2]
        
        self.mapping_data = mapping_data
    
    def _extract_si_parameters(self):
        """提取 SI 相关参数。"""
        # 提取谱图处理参数
        si_feature = self.get_path(self.si_feature_path)
        spectrum_image = self.get_path(
            self.get_path(si_feature['eds']['spectrumImage'])['dataPath']
        )
        
        spectrum_image_settings = bytes_to_json(spectrum_image['SpectrumImageSettings'])
        quant_settings = self.get_path(si_feature['eds']['quantificationSettings'])
        background_correction = self.get_path(quant_settings['backgroundCorrection'])
        
        # 处理背景校正中的路径
        for key, value in background_correction.items():
            if isinstance(value, str) and value[0] == '/':
                background_correction[key] = self.get_path(value)
        
        parameters = {
            'si_absorption_settings': quant_settings['absorptionCorrection'],
            'si_background_correction': background_correction,
            'si_element_selected': [
                quant_settings['elementProperties'][num]
                for num in quant_settings['elementSelection']
            ],
            'si_ionization_cross_section_model': quant_settings['ionizationCrossSectionModel'],
            'si_spectra_filter_settings': self.get_path(si_feature['eds']['spectralFiltersettings']),
        }
        
        # 从元数据中提取参数
        if self.mapping_data:
            sample_key = next(iter(self.mapping_data.keys()))
            metadata = self.mapping_data[sample_key]['metadata']
            
            parameters.update({
                'channels': self.spectra_data['total'].shape[0],
                'quantification_mode': si_feature['quantificationMode'],
                'dwelltime': float(metadata['Scan']['DwellTime']),
                'pixelsize': self._get_pixel_size(metadata)[0],
                'pixelunit': self._get_pixel_size(metadata)[1],
            })
            
            # 提取探测器信息
            for key, value in metadata['Detectors'].items():
                if value.get('RealTime'):
                    parameters[value.get('DetectorName')] = {
                        "RealTime": value.get('RealTime'),
                        "LiveTime": value.get('LiveTime'),
                        "InputCountRate": value.get('InputCountRate'),
                        "OutputCountRate": value.get('OutputCountRate'),
                    }
            
            if 'Detectors' in metadata:
                detector_values = list(metadata['Detectors'].values())
                if detector_values:
                    first_detector = detector_values[-1]
                    parameters['OffsetEnergy'] = float(first_detector.get('OffsetEnergy'))
                    parameters['BeginEnergy'] = float(first_detector.get('BeginEnergy'))
        
        self._update_parameters(parameters)
    
    # ========================================================================
    # 积分谱图处理
    # ========================================================================
    
    def extract_integrated_spectra(self):
        """提取积分谱图。"""
        integrated_spectra_feature = self.get_path(self.integrated_spectra_feature_path)
        quant_settings = self.get_path(integrated_spectra_feature['quantificationSettings'])
        
        # 处理背景校正中的路径
        background_correction = self.get_path(quant_settings['backgroundCorrection'])
        for key, value in background_correction.items():
            if isinstance(value, str) and value[0] == '/':
                background_correction[key] = self.get_path(value)
        
        parameters = {
            'spectra_absorption_settings': quant_settings['absorptionCorrection'],
            'spectra_background_correction': background_correction,
            'spectra_background_windows': self.get_path(
                quant_settings['backgroundWindows']
            )['backgroundWindows'],
            'spectra_element_selected': [
                quant_settings['elementProperties'][num]
                for num in quant_settings['elementSelection']
            ],
            'spectra_ionization_cross_section_model': quant_settings['ionizationCrossSectionModel'],
        }
        
        # 提取谱图数据（如果尚未提取）
        if not hasattr(self, 'spectra_data'):
            self._extract_spectrum_data()
        
        self._update_parameters(parameters)
        return self
    
    def _extract_spectrum_data(self):
        """从 HDF5 文件中提取谱图数据。"""
        spectra_data = {}
        
        if 'Spectrum' in self.f['Data']:
            spectrum_groups = self.f['Data']['Spectrum'].keys()
            channels = self.parameters.get('channels', 4096)
            data_sum = np.zeros(channels, dtype=float)
            
            for spec_id in spectrum_groups:
                spec_path = f'Data/Spectrum/{spec_id}/Data'
                metadata_path = f'Data/Spectrum/{spec_id}/Metadata'
                
                spectrum = self.f[spec_path][:, 0]
                metadata = decode_metadata(self.f[metadata_path])
                data_sum += spectrum
                
                spectra_data[metadata['BinaryResult']['Detector']] = {
                    'spectrum': spectrum,
                    'metadata': metadata,
                }
            
            spectra_data['total'] = data_sum
        
        self.spectra_data = spectra_data
    
    # ========================================================================
    # 线剖面处理
    # ========================================================================
    
    def get_line_profile(self):
        """
        根据线宽提取剖面数据（收集中心线两侧的数据）。
        
        说明:
            这个方法会沿着中心线，在指定宽度范围内提取数据，
            对于每个采样点，会提取垂直于中心线方向上的多个像素点。
        """
        # 获取图像尺寸
        if not hasattr(self, 'mapping_data'):
            self.get_color_mix_image()
        
        image_shape = self.parameters.get('image_shape')
        if not image_shape:
            print("[ERROR] 无法从 self.parameters 获取图像尺寸")
            return self
        
        # 获取线坐标信息
        line_data = self._extract_line_data()
        if not line_data:
            return self
        
        # 转换为像素坐标
        height, width = image_shape
        start = (
            int(line_data['start'][0] * width),
            int(line_data['start'][1] * height)
        )
        end = (
            int(line_data['end'][0] * width),
            int(line_data['end'][1] * height)
        )
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        line_length = np.sqrt(dx**2 + dy**2)
        
        line_position = {
            'start': start,
            'end': end,
            'line_width': line_data['line_width'],
            'dx': dx,
            'dy': dy,
            'length': line_length,
        }
        
        # 计算采样点
        sample_positions, perpendicular_samples = self._calculate_sample_points(line_position)
        
        print(f"每个采样点收集 {line_data['line_width']} 个横向像素点")
        
        # 为每个显示组件提取数据
        profile_data_with_width = self._extract_profile_data(
            sample_positions,
            perpendicular_samples
        )
        
        self.line_position = line_position
        self.line_profile_data = profile_data_with_width
        
        return self
    
    def _extract_line_data(self):
        """从特征中提取线数据。"""
        if not hasattr(self, 'color_mix_profile_feature_path'):
            print("[ERROR] 没有颜色混合特征")
            return None
        
        feature_json = self.get_path(self.color_mix_profile_feature_path)
        annotation_path = feature_json.get('annotation', '')
        annotation_json = self.get_path(annotation_path)
        shape_path = annotation_json.get('shape', '')
        shape_json = self.get_path(shape_path)
        
        line_data = {}
        
        # 提取线段信息
        if 'line' in shape_json:
            line_info = shape_json['line']
            start = line_info['p1']
            end = line_info['p2']
            line_data['start'] = (start['x'], start['y'])
            line_data['end'] = (end['x'], end['y'])
        
        # 提取线宽
        appearance_path = annotation_json.get('appearance', '')
        appearance_json = self.get_path(appearance_path)
        line_data['line_width'] = int(appearance_json['lineSettings']['width'])
        
        if 'start' not in line_data:
            print(f"[ERROR] {shape_path} 中没有 'line': {shape_json.keys()}")
            return None
        
        return line_data
    
    def _calculate_sample_points(self, line_position):
        """计算采样点位置。"""
        start = line_position['start']
        dx = line_position['dx']
        dy = line_position['dy']
        line_length = line_position['length']
        line_width = line_position['line_width']
        
        # 计算单位方向向量
        dir_x = dx / line_length
        dir_y = dy / line_length
        
        # 计算单位法向量（垂直于线方向）
        norm_x = -dir_y
        norm_y = dir_x
        
        # 沿线采样点
        num_samples = max(10, int(line_length))
        sample_positions = []
        perpendicular_samples = []
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            
            # 中心点坐标
            center_x = start[0] + t * dx
            center_y = start[1] + t * dy
            sample_positions.append((center_x, center_y))
            
            # 在垂直于线方向上的采样点
            perp_samples = []
            for w in [i - (line_width - 1) / 2 for i in range(line_width)]:
                offset_x = w * norm_x
                offset_y = w * norm_y
                sample_x = center_x + offset_x
                sample_y = center_y + offset_y
                perp_samples.append((sample_x, sample_y, w))  # w 表示距中心线的距离
            
            perpendicular_samples.append(perp_samples)
        
        return sample_positions, perpendicular_samples
    
    def _extract_profile_data(self, sample_positions, perpendicular_samples):
        """提取剖面数据。"""
        def _spline(arr, x, y):
            """双线性插值。"""
            # 创建索引坐标
            x_indices = np.arange(arr.shape[1])  # 列索引
            y_indices = np.arange(arr.shape[0])  # 行索引
            
            # 创建双线性插值器
            spline = RectBivariateSpline(x_indices, y_indices, arr.T, kx=1, ky=1)
            result = spline(x, y)
            
            return float(result.squeeze())
        
        profile_data_with_width = {}
        
        for key, comp in self.mapping_data.items():
            comp_id = key
            frame_index = comp.get('frameIndex', 0)
            image_data = comp['data'][:, :, frame_index]
            
            if image_data is not None:
                # 提取每个采样点的垂直剖面
                profile_2d = []  # 2D数组：[沿中心线位置] × [垂直位置]
                
                for perp_line in perpendicular_samples:
                    line_values = []
                    for x, y, w in perp_line:
                        value = _spline(image_data, x, y)
                        line_values.append(value)
                    profile_2d.append(line_values)
                
                # 计算平均剖面（沿垂直方向平均）
                profile_avg = np.mean(profile_2d, axis=1) if profile_2d else np.array([])
                
                profile_data_with_width[comp_id] = {
                    'profile_2d': np.array(profile_2d),  # 2D剖面数据
                    'profile_avg': profile_avg,  # 平均剖面
                    'color': comp['color']
                }
                
                print(f"✓ {comp_id}: 提取了 {len(profile_2d)}×{len(profile_2d[0]) if profile_2d else 0} 的 2D 剖面")
        
        return profile_data_with_width
    
    # ========================================================================
    # 颜色混合图像处理
    # ========================================================================

    def select_color_mix_image(self):
        """生成颜色混合图像 (RGB 格式)。"""        
        # 提取映射数据
        if not hasattr(self, 'mapping_data'):
            if not hasattr(self, 'color_mix_profile_feature_path'):
                print('[ERROR] 没有颜色混合特征')
                return self
            else:
                self._extract_color_mix_mapping_data()
        
        # 创建空的 RGB 图像
        height, width = self.parameters['image_shape']
        color_mix_image = np.zeros((height, width, 3), dtype=np.float32)
        
        # 混合所有组件
        for comp_id, comp_data in self.mapping_data.items():
            selected = input(f"Color Mix 中是否要显示 {comp_id}：(Y/N) \n").strip().lower()
            if 'y' in selected:
                color_mix_image = self._blend_component(
                    color_mix_image,
                    comp_data,
                    height,
                    width
                )
            # fig, ax = plt.subplots(figsize=(10, 10))
            # ax.imshow(color_mix_image, origin='upper', aspect='equal')
            # plt.show()
        
        # 归一化到 0-1 范围
        # if color_mix_image.max() > 0:
            # color_mix_image = color_mix_image / color_mix_image.max()
        
        color_mix_image = np.clip(color_mix_image, 0, 1)
        self.selected_color_mix_image = color_mix_image
        
        return self

    def get_color_mix_image(self):
        """生成颜色混合图像 (RGB 格式)。"""        
        # 提取映射数据
        if not hasattr(self, 'mapping_data'):
            if not hasattr(self, 'color_mix_profile_feature_path'):
                print('[ERROR] 没有颜色混合特征')
                return self
            else:
                self._extract_color_mix_mapping_data()
        
        # 创建空的 RGB 图像
        height, width = self.parameters['image_shape']
        color_mix_image = np.zeros((height, width, 3), dtype=np.float32)
        
        # 混合所有组件
        for comp_id, comp_data in self.mapping_data.items():
            color_mix_image = self._blend_component(
                color_mix_image,
                comp_data,
                height,
                width
            )
        
        # 归一化到 0-1 范围
        if color_mix_image.max() > 0:
            color_mix_image = color_mix_image / color_mix_image.max()
        
        color_mix_image = np.clip(color_mix_image, 0, 1)
        self.color_mix_image = color_mix_image
        
        return self
    
    def _extract_color_mix_mapping_data(self):
        """提取颜色混合的映射数据。"""
        color_mix_profile_feature = self.get_path(self.color_mix_profile_feature_path)
        color_mix_profile_inputdata = self.get_path(
            color_mix_profile_feature['imageInputData']
        )
        
        color_mix_display_group_items = [
            self.get_path(item) for item in color_mix_profile_inputdata['displayGroupItems']
        ]
        
        mapping_data = {}
        parameters = {}
        
        for item in color_mix_display_group_items:
            display = self.get_path(item['display'])
            data_obj = self.get_path(display['data'])
            data_path = data_obj['dataPath']
            
            data_item = self.get_path(data_path)
            
            mapping_data[display['id']] = {
                'data': data_item['Data'][()],
                'metadata': decode_metadata(data_item['Metadata']),
                'frame_index': int(data_obj['frameIndex']),
                'color': self.get_path(display['settings'])['color'],
                'display_range': [
                    float(num) for num in self.get_path(display['settings'])['displayLevelsRange'].values()
                ],
                'gamma': float(self.get_path(display['settings'])['gamma']),
                'blend_factor': item['blendFactor'],
                'blend_mode': item['blendMode'],
            }
            
            # 如果是 STEM 图像，提取参数
            if item['groupType'] == 'Stem':
                metadata = decode_metadata(data_item['Metadata'])
                data = data_item['Data'][()]
                
                parameters['frames'] = int(
                    metadata['CustomProperties']['Velox.Series.FrameNumber']['value']
                )
                parameters['image_shape'] = data.shape[:2]
        
        self.mapping_data = mapping_data
        self._update_parameters(parameters)
    
    def _blend_component(self, base_image, comp_data, height, width):
        """混合单个组件到基础图像中。"""
        frame_index = comp_data['frame_index']
        image = comp_data['data'][:, :, frame_index]
        blend_factor = comp_data['blend_factor']
        blend_mode = comp_data['blend_mode']
        
        color_dict = comp_data['color']
        color = np.array([color_dict['red'], color_dict['green'], color_dict['blue']])
        
        display_range = comp_data['display_range']
        gamma = comp_data['gamma']
        
        # 1. 应用显示范围
        begin, end = display_range
        if end > begin:
            image_scaled = (image - begin) / (end - begin)
        else:
            image_scaled = (image - end) / (begin - end)
        
        image_scaled = np.clip(image_scaled, 0, 1)
        
        # 2. 应用 Gamma 校正
        if gamma != 1.0:
            image_scaled = np.power(image_scaled, gamma)
        
        # 3. 应用混合因子
        intensity = image_scaled * blend_factor
        
        # 4. 创建彩色层
        color_layer = np.zeros((height, width, 3), dtype=np.float32)
        for c in range(3):
            color_layer[:, :, c] = intensity * color[c]
        
        # 5. 根据混合模式合并
        if blend_mode == 'Additive':
            return base_image + color_layer
        elif blend_mode == 'Alpha':
            alpha = blend_factor
            return base_image * (1 - alpha) + color_layer * alpha
        elif blend_mode == 'Multiply':
            return base_image * (color_layer + 1e-10)  # 避免乘以0
        elif blend_mode == 'Screen':
            return 1 - (1 - base_image) * (1 - color_layer)
        elif blend_mode == 'Overlay':
            mask = base_image > 0.5
            return np.where(
                mask,
                1 - 2 * (1 - base_image) * (1 - color_layer),
                2 * base_image * color_layer
            )
        else:
            return base_image + color_layer
    
    # ========================================================================
    # 裁剪图像处理
    # ========================================================================
    
    def get_crop_image(self):
        """获取裁剪后的图像。"""
        if not hasattr(self, 'crop_feature_path'):
            print('[ERROR] 没有裁剪图像')
            return self
        
        crop_feature = self.get_path(self.crop_feature_path)
        crop_image_display = self.get_path(crop_feature['imageDisplay'])
        
        series_index = int(crop_image_display['seriesIndex'])
        data_obj = self.get_path(crop_image_display['dataPath'])
        
        crop_image_data = data_obj['Data'][:, :, :]
        crop_image_metadata = decode_metadata(data_obj['Metadata'])
        
        crop_annotation = self.get_path(crop_feature['cropAnnotationPath'])
        crop_annotation_data = self.get_path(crop_annotation['dataPath'])
        parameters = {}
        parameters['crop'] = {
            'image_name': f"{unquote(crop_image_display['display']['label'])} of {Path(self.file_path).stem}",
            'pixelsize': self._get_pixel_size(crop_image_metadata)[0],
            'pixelunit': self._get_pixel_size(crop_image_metadata)[1],
            'display_index': series_index,
            'display_range': [
                float(num) for num in crop_image_display['displayLevelsRange'].values()
            ],
            'gamma': float(crop_image_display['gamma']),
            'annotation_shape': crop_annotation_data,
            'annotation_color': crop_annotation['color'],
        }
        
        self.crop_data = crop_image_data
        self.crop_metadata = crop_image_metadata
        self._update_parameters(parameters)
        return self
    
    # ========================================================================
    # 滤波图像处理
    # ========================================================================
    
    def get_filtered_image(self):
        """获取滤波后的图像。"""
        if not hasattr(self, 'image_filter_feature_path'):
            print('[ERROR] 没有滤波图像')
            return self
        
        filter_feature = self.get_path(self.image_filter_feature_path)
        filtered_image_display = self.get_path(filter_feature['imageDisplay'])
        
        series_index = int(filtered_image_display['seriesIndex'])
        data_obj = self.get_path(filtered_image_display['dataPath'])
        
        filtered_image_data = data_obj['Data'][:, :, :]
        filtered_image_metadata = decode_metadata(data_obj['Metadata'])
        
        filtered_operation = self.get_path(filter_feature['imageFilteringOperationRecord'])
        filter_settings = self.get_path(filtered_operation['settingsPath'])
        filter_type = filtered_operation['filterType']

        parameters = {}
        parameters['filter'] = {
            'image_name': f"{unquote(filtered_image_display['display']['label'])} of {Path(self.file_path).stem}",
            'pixelsize': self._get_pixel_size(filtered_image_metadata)[0],
            'pixelunit': self._get_pixel_size(filtered_image_metadata)[1],
            'display_index': series_index,
            'display_range': [
                float(num) for num in filtered_image_display['displayLevelsRange'].values()
            ],
            'gamma': float(filtered_image_display['gamma']),
            'filter_settings': filter_settings,
            'filter_type': filter_type,
        }
        
        self.filter_data = filtered_image_data
        self.filter_metadata = filtered_image_metadata
        self._update_parameters(parameters)
        return self

    # ========================================================================
    # 数据展示
    # ========================================================================
    
    def display(self):
        """根据检测到的特征类型执行相应的数据导出方法。"""
        feature_handlers = [
            ('si_feature_path', self.display_si),
            ('integrated_spectra_feature_path', self.display_integrated_spectra),
            ('color_mix_profile_feature_path', self.display_color_mix_and_line_profile),
            ('camera_feature_path', self.display_tem_image),
            ('dcfi_feature_path', self.display_dcfi_image),
            ('stem_feature_path', self.display_stem_image),
            ('dpc_feature_path', self.display_dpc_images),
            ('crop_feature_path', self.display_crop_image),
            ('image_filter_feature_path', self.display_filtered_image)
        ]
        
        for feature_attr, handler in feature_handlers:
            if hasattr(self, feature_attr):
                handler()

    def display_si(self):
        """先一张一张展示，后续改为统一展示"""
        for key, value in self.mapping_data.items():
            print(f"Color of {key}: {value['color']}")
            cmap = LinearSegmentedColormap.from_list('custom_cmap', colors=[(0,0,0),(value['color']['red'], value['color']['green'], value['color']['blue'])], N=256)
            fig, axe = display_image_with_scale(image = value['data'], pixel_size = self.parameters['pixelsize'], pixel_unit = self.parameters['pixelunit'], title = f"{key}: {value['color']}", cmap = cmap, display_range = value['display_range'], gamma = value['gamma'], display_index = value['frame_index'])
            plt.show()

    def display_integrated_spectra(self):
        """先一张一张展示，后续改为统一展示"""
        fig = plot_spectrum(self.spectra_data['total'], start = self.parameters['OffsetEnergy']+2.5, end = 4095*5+self.parameters['OffsetEnergy']+2.5, log = False, title = "Energy Spectrum", xlabel = "Energy (eV)", ylabel = "Counts", figsize = (7, 4), save_path = None, dpi = 300, show = False, grid = True, minor_ticks = True, color = "steelblue", lw = 0.8)
        plt.show()
    
    def display_color_mix_and_line_profile(self):
        """先一张一张展示，后续改为统一展示"""
        fig, ax = draw_line_annotation_on_image(self.color_mix_image, line_info=self.line_position, pixel_size=self.parameters['pixelsize'], pixel_unit=self.parameters['pixelunit'])
        plt.show()
        fig = draw_line_profiles(self.line_profile_data, pixel_size=self.parameters['pixelsize'], pixel_unit=self.parameters['pixelunit'])
        plt.show()
    
    def display_tem_image(self):
        """先一张一张展示，后续改为统一展示"""
        for key, value in self.parameters.items():
            if key == 'Ceta':
                print(f"图像形状（{key}）: {self.tem_data.shape}")
                fig, axe = display_image_with_scale(image = self.tem_data, pixel_size = value['pixelsize'], pixel_unit = value['pixelunit'], title = value['image_name'], cmap = 'gray', display_range = value['display_range'], gamma = value['gamma'], display_index = value['display_index'])
                plt.show()
    
    def display_dcfi_image(self):
        """先一张一张展示，后续改为统一展示"""
        for key, value in self.parameters.items():
            if key == 'DCFI':
                print(f"图像形状（{key}）: {self.dcfi_data.shape}")
                fig, axe = display_image_with_scale(image = self.dcfi_data, pixel_size = value['pixelsize'], pixel_unit = value['pixelunit'], title = value['image_name'], cmap = 'gray', display_range = value['display_range'], gamma = value['gamma'], display_index = value['display_index'])
                plt.show()
    
    def display_stem_image(self):
        """先一张一张展示，后续改为统一展示"""
        for key, value in self.parameters.items():
            if key in self.stem_data:
                print(f"图像形状（{key}）: {self.stem_data[key].shape}")
                fig, axe = display_image_with_scale(image = self.stem_data[key], pixel_size = value['pixelsize'], pixel_unit = value['pixelunit'], title = value['image_name'], cmap = 'gray', display_range = value['display_range'], gamma = value['gamma'], display_index = value['display_index'])
                plt.show()
    
    def display_dpc_images(self):
        """先一张一张展示，后续改为统一展示"""
        for key, value in self.parameters.items():
            if key in self.dpc_data:
                print(f"图像形状（{key}）: {self.dpc_data[key].shape}")
                fig, axe = display_image_with_scale(image = self.dpc_data[key], pixel_size = value['pixelsize'], pixel_unit = value['pixelunit'], title = value['image_name'], cmap = 'gray', display_range = value['display_range'], gamma = value['gamma'], display_index = value['display_index'])
                plt.show()
    
    def display_crop_image(self):
        """先一张一张展示，后续改为统一展示"""
        key = 'crop'
        value = self.parameters[key]
        fig, axe = display_image_with_scale(image = self.crop_data, pixel_size = value['pixelsize'], pixel_unit = value['pixelunit'], title = value['image_name'], cmap = 'gray', display_range = value['display_range'], gamma = value['gamma'], display_index = value['display_index'])
        plt.show()
    
    def display_filtered_image(self):
        """先一张一张展示，后续改为统一展示"""
        key = 'filter'
        value = self.parameters[key]
        fig, axe = display_image_with_scale(image = self.crop_data, pixel_size = value['pixelsize'], pixel_unit = value['pixelunit'], title = value['image_name'], cmap = 'gray', display_range = value['display_range'], gamma = value['gamma'], display_index = value['display_index'])
        plt.show()

    # ========================================================================
    # 数据导出
    # ========================================================================
    
    def export(self, export_type=None):
        """根据检测到的特征类型执行相应的数据导出方法。"""
        if export_type is None:
            export_type = {
                "image_export_type": "2",
                "colormix_export_type": "y",
                "eds_export_type": "y",
                "quanti_export_type": "y",
                "lineprofile_export_type": "1",
            }
        
        feature_handlers = [
            ('si_feature_path', self.export_si),
            ('integrated_spectra_feature_path', self.export_integrated_spectra),
            ('color_mix_profile_feature_path', self.export_color_mix_and_line_profile),
            ('camera_feature_path', self.export_tem_image),
            ('dcfi_feature_path', self.export_dcfi_image),
            ('stem_feature_path', self.export_stem_image),
            ('dpc_feature_path', self.export_dpc_images),
            ('crop_feature_path', self.export_crop_image),
            ('image_filter_feature_path', self.export_filtered_image)
        ]
        
        for feature_attr, handler in feature_handlers:
            if hasattr(self, feature_attr):
                handler(export_type)

    def check_output_dir(self):
        # 确保输出路径存在 r".\custom_export\{origin_filename_stem}\"
         origin_path = Path(self.file_path)
         origin_filename_stem = origin_path.stem
         target_dir = origin_path.parent / "custom_export" / origin_filename_stem
         target_dir.mkdir(parents=True, exist_ok=True)
         self.export_dir = target_dir
         return self

    def export_si(self, export_type=''):
        """
        将 STEM 和 Elemental mapping 导出
        导出为 16bit TIFF (data) 和/或 png (with scale bar)
        """
        export_option = input("""STEM图像和元素分布图导出格式（可以输入单独的数字或者组合）：\n  1. All\n  2. Data (DM5)\n  3. Data (16-bit TIFF)\n  4. Image (PNG)\n  5. Quantification Result (CSV)\n""").strip()
        quantification_mode = self.parameters['quantification_mode']
        filename_stem = Path(self.file_path).stem
        # 确保输出路径存在 r".\custom_export\{origin_filename_stem}\"
        self.check_output_dir()
        if '1' in export_option or '5' in export_option:
            filename = f"{filename_stem}-Quatification-{quantification_mode}.csv"
            output_path = self.export_dir / filename
            try:
                html_table_to_csv(self.experiment_log, output_path = str(output_path))
            except Exception as e:
                print(f"导出 Quantification Result 出错：{e}")
        if '1' in export_option or '4' in export_option:
            if hasattr(self, 'color_mix_image'):
                export_color_mix = input("""是否导出 Color Mix Image（无 Annotation，PNG 格式）： (Y/N)\n""").strip().lower()
                if 'y' in export_color_mix:
                    answer = input("""Color Mix 中是否需要所有的STEM图像和元素分布图： (Y/N)\n""").strip().lower()
                    if 'y' in answer:
                        filename = f"{filename_stem}-Colormix.png"
                        output_path = self.export_dir / filename
                        save_color_mix_image(
                            image = self.color_mix_image,
                            output_path = output_path,
                            pixel_size = self.parameters['pixelsize'],
                            pixel_unit = self.parameters['pixelunit']
                        )
                    else:
                        if not hasattr(self, 'selected_color_mix_image'):
                            self.select_color_mix_image()
                        filename = f"{filename_stem}-SelectColormix.png"
                        output_path = self.export_dir / filename
                        save_color_mix_image(
                            image = self.selected_color_mix_image,
                            output_path = output_path,
                            pixel_size = self.parameters['pixelsize'],
                            pixel_unit = self.parameters['pixelunit']
                        )
        for key, value in self.mapping_data.items():
            filename = f"{filename_stem}-{key}-{quantification_mode}"
            data = value['data'][:,:,value['frame_index']] # elemental mapping 一般都只有一帧
            output_path = self.export_dir / filename
            if '1' in export_option or '2' in export_option:
                dm5_writer(add_suffix_safe(output_path, '.dm5'), value, self.parameters)
            if '1' in export_option or '3' in export_option:
                imagej_metadata = {}
                imagej_metadata['ImageJ'] = '1.54g'
                if data.ndim == 3:
                    imagej_metadata['slices'] = data.shape[-1] # (height, width, slices)？
                imagej_metadata['unit'] = self.parameters['pixelunit'].replace('μ', 'u') # 替换非ASCII字符
                imagej_metadata['spacing'] = self.parameters['pixelsize']
                save_as_16bit_tiff(data, output_path, metadata=imagej_metadata)
            if '1' in export_option or '4' in export_option:
                color_rgb = (value['color']['red'], value['color']['green'], value['color']['blue'])
                cmap = LinearSegmentedColormap.from_list(
                    'custom_cmap', 
                    colors=[(0, 0, 0), color_rgb], 
                    N=256
                )
                success = save_image_as_png(
                    image=data,
                    output_path=add_suffix_safe(output_path, '.png'),
                    pixel_size=self.parameters['pixelsize'],
                    pixel_unit=self.parameters['pixelunit'],
                    cmap=cmap,
                    display_range=value['display_range'],
                    gamma=value['gamma'],
                    display_index=value['frame_index'],
                    add_scalebar=True
                )
        return self
            
    def export_integrated_spectra(self):
        """
        将 eds spectra 导出
        导出为 csv 文件
        """
        export_option = input("""是否需要导出 EDS Spectra: (Y/N)\n""").strip().lower()
        if not "y" in export_option:
            return self
        # 确保输出路径存在 r".\custom_export\{origin_filename_stem}\"
        self.check_output_dir()
        filename_stem = Path(self.file_path).stem
        filename = f"{filename_stem}-Spectra.csv"
        output_path = self.export_dir / filename
        export_eds_spectrum(output_path=output_path, intensity=self.spectra_data['total'], offset=self.parameters['OffsetEnergy'], channels=4096, dispension=5)
        return self
    
    def export_color_mix_and_line_profile(self):
        # 如果一个文件只有 color mix，没有 line profile 怎么导出？
        # draw_line_annotation_on_image 这个函数不传递 line_info 就可以画出没有 line annotation 的 color mix
        # 如果想要选择不同的元素分布图，做出不同的 color mix 该怎么做？
        quantification_mode = self.parameters['quantification_mode']
        filename_stem = Path(self.file_path).stem
        # 确保输出路径存在 r".\custom_export\{origin_filename_stem}\"
        self.check_output_dir()
        export_option = input("""Color Mix Image 和 Line Profiles 导出格式：\n  1. All\n  2. Data (CSV)\n  3. Image (PNG)\n""").strip()
        if '1' in export_option or '3' in export_option:
            answer = input("""Color Mix with Annotation 中是否需要所有的STEM图像和元素分布图： (Y/N)\n""").strip().lower()
            if 'y' in answer:
                filename = f"{filename_stem}-Colormix-LineAnnotation.png"
                output_path = self.export_dir / filename
                save_color_mix_image(self.color_mix_image, output_path = output_path, line_info=self.line_position, pixel_size=self.parameters['pixelsize'], pixel_unit=self.parameters['pixelunit'])
            else:
                if not hasattr(self, 'selected_color_mix_image'):
                    self.select_color_mix_image()
                filename = f"{filename_stem}-SelectColomix-LineAnnotation.png"
                output_path = self.export_dir / filename
                save_color_mix_image(
                    image = self.selected_color_mix_image,
                    output_path = output_path,
                    line_info=self.line_position, 
                    pixel_size = self.parameters['pixelsize'],
                    pixel_unit = self.parameters['pixelunit']
                )

            filename = f"{filename_stem}-Colormix-LineProfile.png"
            output_path = self.export_dir / filename
            fig = draw_line_profiles(self.line_profile_data, output_path = output_path, pixel_size=self.parameters['pixelsize'], pixel_unit=self.parameters['pixelunit'])
        if '1' in export_option or '2' in export_option:
            filename = f"{filename_stem}-Colormix-LineProfile.csv"
            output_path = self.export_dir / filename
            export_line_profile_as_csv(output_path, line_length_pixel=self.line_position['length'], line_profile=self.line_profile_data, pixelsize=self.parameters['pixelsize'], pixelunit=self.parameters['pixelunit'], quantification_mode=quantification_mode)
    
    def export_tem_image(self):
        """
        将 TEM 图像导出
        导出为 16bit TIFF (data) 和/或 png (with scale bar)
        """
        export_option = input("""TEM图像导出格式（可以输入单独的数字或者组合）：\n  1. All\n  2. Data (DM5)\n  3. Data (16-bit TIFF)\n  4. Image (PNG)\n""").strip()
        filename_stem = Path(self.file_path).stem
        # 确保输出路径存在 r".\custom_export\{origin_filename_stem}\"
        self.check_output_dir()
        if '1' in export_option or '2' in export_option:
            filename = f"{filename_stem}.dm5"
            output_path = self.export_dir / filename
            signal = {}
            signal['data'] = self.tem_data
            if self.tem_data.ndim == 2:
                signal['data'] = self.tem_data[..., np.newaxis]
            signal['metadata'] = self.tem_metadata
            signal['color'] = {'blue': 1, 'green': 1, 'red': 1}
            signal['display_range'] = self.parameters['Ceta']['display_range']
            dm5_writer(output_path, signal, self.parameters['Ceta'])
        if '1' in export_option or '3' in export_option:
            filename = f"{filename_stem}.tif"
            output_path = self.export_dir / filename
            imagej_metadata = {}
            imagej_metadata['ImageJ'] = '1.54g'
            data = self.tem_data
            if data.ndim == 3:
                imagej_metadata['slices'] = data.shape[-1] # (height, width, slices)？
            imagej_metadata['unit'] = self.parameters['Ceta']['pixelunit'].replace('μ', 'u') # 替换非ASCII字符
            imagej_metadata['spacing'] = self.parameters['Ceta']['pixelsize']
            save_as_16bit_tiff(data, output_path, metadata=imagej_metadata)
        if '1' in export_option or '4' in export_option:
            filename = f"{filename_stem}.png"
            output_path = self.export_dir / filename
            parameters = self.parameters['Ceta']
            if self.tem_data.ndim == 2:
                # 如果是单张图像
                success = save_image_as_png(
                    image=self.tem_data,
                    output_path=output_path,
                    pixel_size=parameters['pixelsize'],
                    pixel_unit=parameters['pixelunit'],
                    # cmap='gray',
                    display_range=parameters['display_range'],
                    gamma=parameters['gamma'],
                    # display_index=parameters['display_index'],
                    add_scalebar=True
                )
            elif self.tem_data.ndim == 3:
                success = save_image_as_png(
                    image=self.tem_data,
                    output_path=output_path,
                    pixel_size=parameters['pixelsize'],
                    pixel_unit=parameters['pixelunit'],
                    # cmap='gray',
                    display_range=parameters['display_range'],
                    gamma=parameters['gamma'],
                    display_index=parameters['display_index'],
                    add_scalebar=True
                )
            else:
                print(f"tem_data 的 shape 有问题，需要检查\nself.tem_data.shape: {self.tem_data.shape}")
        return self
    
    def export_dcfi_image(self):
        """
        将 DCFI 图像导出
        导出为 16bit TIFF (data) 和/或 png (with scale bar)
        """
        export_option = input("""DFCI图像导出格式（可以输入单独的数字或者组合）：\n  1. All\n  2. Data (DM5)\n  3. Data (16-bit TIFF)\n  4. Image (PNG)\n""").strip()
        # filename_stem = Path(self.file_path).stem
        assert 'DCFI' in self.parameters.keys()
        filename_stem = self.parameters['DCFI']['image_name']
        # 确保输出路径存在 r".\custom_export\{origin_filename_stem}\"
        self.check_output_dir()
        if '1' in export_option or '2' in export_option:
            filename = f"{filename_stem}.dm5"
            output_path = self.export_dir / filename
            signal = {}
            signal['data'] = self.dcfi_data
            if self.dcfi_data.ndim == 2:
                signal['data'] = self.dcfi_data[..., np.newaxis]
            signal['metadata'] = self.dcfi_metadata
            signal['color'] = {'blue': 1, 'green': 1, 'red': 1}
            signal['display_range'] = self.parameters['DCFI']['display_range']
            dm5_writer(output_path, signal, self.parameters['DCFI'])
        if '1' in export_option or '3' in export_option:
            filename = f"{filename_stem}.tif"
            output_path = self.export_dir / filename
            imagej_metadata = {}
            imagej_metadata['ImageJ'] = '1.54g'
            data = self.dcfi_data
            if data.ndim == 3:
                imagej_metadata['slices'] = data.shape[-1] # (height, width, slices)？
            imagej_metadata['unit'] = self.parameters['DCFI']['pixelunit'].replace('μ', 'u') # 替换非ASCII字符
            imagej_metadata['spacing'] = self.parameters['DCFI']['pixelsize']
            save_as_16bit_tiff(data, output_path, metadata=imagej_metadata)
        if '1' in export_option or '4' in export_option:
            filename = f"{filename_stem}.png"
            output_path = self.export_dir / filename
            parameters = self.parameters['DCFI']
            if self.dcfi_data.ndim == 2:
                # 如果是单张图像
                success = save_image_as_png(
                    image=self.dcfi_data,
                    output_path=output_path,
                    pixel_size=parameters['pixelsize'],
                    pixel_unit=parameters['pixelunit'],
                    # cmap='gray',
                    display_range=parameters['display_range'],
                    gamma=parameters['gamma'],
                    # display_index=parameters['display_index'],
                    add_scalebar=True
                )
            elif self.dcfi_data.ndim == 3:
                success = save_image_as_png(
                    image=self.dcfi_data,
                    output_path=output_path,
                    pixel_size=parameters['pixelsize'],
                    pixel_unit=parameters['pixelunit'],
                    # cmap='gray',
                    display_range=parameters['display_range'],
                    gamma=parameters['gamma'],
                    display_index=parameters['display_index'],
                    add_scalebar=True
                )
            else:
                print(f"dcfi_data 的 shape 有问题，需要检查\nself.dcfi_data.shape: {self.dcfi_data.shape}")
        return self
    
    def export_stem_image(self):
        """
        将 STEM 图像导出
        导出为 16bit TIFF (data) 和/或 png (with scale bar)
        """
        export_option = input("""STEM图像导出格式（可以输入单独的数字或者组合）：\n  1. All\n  2. Data (DM5)\n  3. Data (16-bit TIFF)\n  4. Image (PNG)\n""").strip()
        # filename_stem = Path(self.file_path).stem
        # 确保输出路径存在 r".\custom_export\{origin_filename_stem}\"
        self.check_output_dir()
        for key, parameters in self.parameters.items():
            try:
                data = self.stem_data[key]
                metadata = self.stem_metadata[key]
            except:
                continue
            if '1' in export_option or '2' in export_option:
                filename = f"{parameters['image_name']}.dm5"
                output_path = self.export_dir / filename
                signal = {}
                signal['data'] = data
                if data.ndim == 2:
                    signal['data'] = data[..., np.newaxis]
                signal['metadata'] = metadata
                signal['color'] = {'blue': 1, 'green': 1, 'red': 1}
                signal['display_range'] = parameters['display_range']
                dm5_writer(output_path, signal, parameters)
            if '1' in export_option or '3' in export_option:
                filename = f"{parameters['image_name']}.tif"
                output_path = self.export_dir / filename
                imagej_metadata = {}
                imagej_metadata['ImageJ'] = '1.54g'
                # data = self.tem_data
                if data.ndim == 3:
                    imagej_metadata['slices'] = data.shape[-1] # (height, width, slices)？
                imagej_metadata['unit'] = parameters['pixelunit'].replace('μ', 'u') # 替换非ASCII字符
                imagej_metadata['spacing'] = parameters['pixelsize']
                save_as_16bit_tiff(data, output_path, metadata=imagej_metadata)
            if '1' in export_option or '4' in export_option:
                filename = f"{parameters['image_name']}.png"
                output_path = self.export_dir / filename
                # parameters = self.parameters['Ceta']
                if data.ndim == 2:
                    # 如果是单张图像
                    success = save_image_as_png(
                        image=data,
                        output_path=output_path,
                        pixel_size=parameters['pixelsize'],
                        pixel_unit=parameters['pixelunit'],
                        # cmap='gray',
                        display_range=parameters['display_range'],
                        gamma=parameters['gamma'],
                        # display_index=parameters['display_index'],
                        add_scalebar=True
                    )
                elif data.ndim == 3:
                    success = save_image_as_png(
                        image=data,
                        output_path=output_path,
                        pixel_size=parameters['pixelsize'],
                        pixel_unit=parameters['pixelunit'],
                        # cmap='gray',
                        display_range=parameters['display_range'],
                        gamma=parameters['gamma'],
                        display_index=parameters['display_index'],
                        add_scalebar=True
                    )
                else:
                    print(f"stem_data 的 shape 有问题，需要检查\nself.stem_data[{key}].shape: {data.shape}")
        return self
    
    def export_dpc_images(self):
        """
        将 DPC 图像导出
        导出为 16bit TIFF (data) 和/或 png (with scale bar)
        """
        export_option = input("""DPC图像导出格式（可以输入单独的数字或者组合）：\n  1. All\n  2. Data (DM5)\n  3. Data (16-bit TIFF)\n  4. Image (PNG)\n""").strip()
        # filename_stem = Path(self.file_path).stem
        # 确保输出路径存在 r".\custom_export\{origin_filename_stem}\"
        self.check_output_dir()
        for key, parameters in self.parameters.items():
            try:
                data = self.dpc_data[key]
                metadata = self.dpc_metadata[key]
            except:
                continue
            if '1' in export_option or '2' in export_option:
                filename = f"{parameters['image_name']}.dm5"
                output_path = self.export_dir / filename
                signal = {}
                signal['data'] = data
                if data.ndim == 2:
                    signal['data'] = data[..., np.newaxis]
                signal['metadata'] = metadata
                signal['color'] = {'blue': 1, 'green': 1, 'red': 1}
                signal['display_range'] = parameters['display_range']
                dm5_writer(output_path, signal, parameters)
            if '1' in export_option or '3' in export_option:
                filename = f"{parameters['image_name']}.tif"
                output_path = self.export_dir / filename
                imagej_metadata = {}
                imagej_metadata['ImageJ'] = '1.54g'
                # data = self.tem_data
                if data.ndim == 3:
                    imagej_metadata['slices'] = data.shape[-1] # (height, width, slices)？
                imagej_metadata['unit'] = parameters['pixelunit'].replace('μ', 'u') # 替换非ASCII字符
                imagej_metadata['spacing'] = parameters['pixelsize']
                save_as_16bit_tiff(data, output_path, metadata=imagej_metadata)
            if '1' in export_option or '4' in export_option:
                filename = f"{parameters['image_name']}.png"
                output_path = self.export_dir / filename
                # parameters = self.parameters['Ceta']
                if data.ndim == 2:
                    # 如果是单张图像
                    success = save_image_as_png(
                        image=data,
                        output_path=output_path,
                        pixel_size=parameters['pixelsize'],
                        pixel_unit=parameters['pixelunit'],
                        # cmap='gray',
                        display_range=parameters['display_range'],
                        gamma=parameters['gamma'],
                        # display_index=parameters['display_index'],
                        add_scalebar=True
                    )
                elif data.ndim == 3:
                    success = save_image_as_png(
                        image=data,
                        output_path=output_path,
                        pixel_size=parameters['pixelsize'],
                        pixel_unit=parameters['pixelunit'],
                        # cmap='gray',
                        display_range=parameters['display_range'],
                        gamma=parameters['gamma'],
                        display_index=parameters['display_index'],
                        add_scalebar=True
                    )
                else:
                    print(f"dpc_data 的 shape 有问题，需要检查\nself.dpc_data[{key}].shape: {data.shape}")
        return self
    
    def export_crop_image(self):
        """
        将 crop 图像导出
        导出为 16bit TIFF (data) 和/或 png (with scale bar)
        """
        export_option = input("""crop图像导出格式（可以输入单独的数字或者组合）：\n  1. All\n  2. Data (DM5)\n  3. Data (16-bit TIFF)\n  4. Image (PNG)\n""").strip()
        # filename_stem = Path(self.file_path).stem
        assert 'crop' in self.parameters.keys()
        filename_stem = self.parameters['crop']['image_name']
        # 确保输出路径存在 r".\custom_export\{origin_filename_stem}\"
        self.check_output_dir()
        if '1' in export_option or '2' in export_option:
            filename = f"{filename_stem}.dm5"
            output_path = self.export_dir / filename
            signal = {}
            signal['data'] = self.crop_data
            if self.crop_data.ndim == 2:
                signal['data'] = self.crop_data[..., np.newaxis]
            signal['metadata'] = self.crop_metadata
            signal['color'] = {'blue': 1, 'green': 1, 'red': 1}
            signal['display_range'] = self.parameters['crop']['display_range']
            dm5_writer(output_path, signal, self.parameters['crop'])
        if '1' in export_option or '3' in export_option:
            filename = f"{filename_stem}.tif"
            output_path = self.export_dir / filename
            imagej_metadata = {}
            imagej_metadata['ImageJ'] = '1.54g'
            data = self.crop_data
            if data.ndim == 3:
                imagej_metadata['slices'] = data.shape[-1] # (height, width, slices)？
            imagej_metadata['unit'] = self.parameters['crop']['pixelunit'].replace('μ', 'u') # 替换非ASCII字符
            imagej_metadata['spacing'] = self.parameters['crop']['pixelsize']
            save_as_16bit_tiff(data, output_path, metadata=imagej_metadata)
        if '1' in export_option or '4' in export_option:
            filename = f"{filename_stem}.png"
            output_path = self.export_dir / filename
            parameters = self.parameters['crop']
            if self.crop_data.ndim == 2:
                # 如果是单张图像
                success = save_image_as_png(
                    image=self.crop_data,
                    output_path=output_path,
                    pixel_size=parameters['pixelsize'],
                    pixel_unit=parameters['pixelunit'],
                    # cmap='gray',
                    display_range=parameters['display_range'],
                    gamma=parameters['gamma'],
                    # display_index=parameters['display_index'],
                    add_scalebar=True
                )
            elif self.crop_data.ndim == 3:
                success = save_image_as_png(
                    image=self.crop_data,
                    output_path=output_path,
                    pixel_size=parameters['pixelsize'],
                    pixel_unit=parameters['pixelunit'],
                    # cmap='gray',
                    display_range=parameters['display_range'],
                    gamma=parameters['gamma'],
                    display_index=parameters['display_index'],
                    add_scalebar=True
                )
            else:
                print(f"crop_data 的 shape 有问题，需要检查\nself.crop_data.shape: {self.crop_data.shape}")
        return self
    
    def export_filtered_image(self):
        """
        将 filter 图像导出
        导出为 16bit TIFF (data) 和/或 png (with scale bar)
        """
        export_option = input("""filter图像导出格式（可以输入单独的数字或者组合）：\n  1. All\n  2. Data (DM5)\n  3. Data (16-bit TIFF)\n  4. Image (PNG)\n""").strip()
        # filename_stem = Path(self.file_path).stem
        assert 'filter' in self.parameters.keys()
        filename_stem = self.parameters['filter']['image_name']
        # 确保输出路径存在 r".\custom_export\{origin_filename_stem}\"
        self.check_output_dir()
        if '1' in export_option or '2' in export_option:
            filename = f"{filename_stem}.dm5"
            output_path = self.export_dir / filename
            signal = {}
            signal['data'] = self.filter_data
            if self.filter_data.ndim == 2:
                signal['data'] = self.filter_data[..., np.newaxis]
            signal['metadata'] = self.filter_metadata
            signal['color'] = {'blue': 1, 'green': 1, 'red': 1}
            signal['display_range'] = self.parameters['filter']['display_range']
            dm5_writer(output_path, signal, self.parameters['filter'])
        if '1' in export_option or '3' in export_option:
            filename = f"{filename_stem}.tif"
            output_path = self.export_dir / filename
            imagej_metadata = {}
            imagej_metadata['ImageJ'] = '1.54g'
            data = self.filter_data
            if data.ndim == 3:
                imagej_metadata['slices'] = data.shape[-1] # (height, width, slices)？
            imagej_metadata['unit'] = self.parameters['filter']['pixelunit'].replace('μ', 'u') # 替换非ASCII字符
            imagej_metadata['spacing'] = self.parameters['filter']['pixelsize']
            save_as_16bit_tiff(data, output_path, metadata=imagej_metadata)
        if '1' in export_option or '4' in export_option:
            filename = f"{filename_stem}.png"
            output_path = self.export_dir / filename
            parameters = self.parameters['filter']
            if self.filter_data.ndim == 2:
                # 如果是单张图像
                success = save_image_as_png(
                    image=self.filter_data,
                    output_path=output_path,
                    pixel_size=parameters['pixelsize'],
                    pixel_unit=parameters['pixelunit'],
                    # cmap='gray',
                    display_range=parameters['display_range'],
                    gamma=parameters['gamma'],
                    # display_index=parameters['display_index'],
                    add_scalebar=True
                )
            elif self.filter_data.ndim == 3:
                success = save_image_as_png(
                    image=self.filter_data,
                    output_path=output_path,
                    pixel_size=parameters['pixelsize'],
                    pixel_unit=parameters['pixelunit'],
                    # cmap='gray',
                    display_range=parameters['display_range'],
                    gamma=parameters['gamma'],
                    display_index=parameters['display_index'],
                    add_scalebar=True
                )
            else:
                print(f"filter_data 的 shape 有问题，需要检查\nself.filter_data.shape: {self.filter_data.shape}")
        return self

    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _get_pixel_size(self, metadata):
        """从元数据中获取像素大小和单位。"""
        pixel_size = float(metadata['BinaryResult']['PixelSize']['width'])
        pixel_unit = metadata['BinaryResult']['PixelUnitX']
        
        # 转换单位
        if metadata['BinaryResult']['PixelUnitX'] == 'm':
            pixel_size *= 1e9
            pixel_unit = 'nm'
            if pixel_size >= 10.0:
                pixel_size /= 1e3
                pixel_unit = 'μm'
        if metadata['BinaryResult']['PixelUnitX'] == '1/m':
            pixel_size /= 1e9
            pixel_unit = '1/nm'
        
        return (pixel_size, pixel_unit)
    
    def _update_parameters(self, new_parameters):
        """更新参数字典。"""
        if hasattr(self, 'parameters'):
            self.parameters.update(new_parameters)
        else:
            self.parameters = new_parameters


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 示例文件路径
    # file_path = r".\emd_files\EDS_LineScan_wt%_20250620_380kx_HAADF.emd"
    # file_path = r".\emd_files\EDS_LineScan_net_20260107_1542_SI_380_kx_HAADF.emd"
    # file_path = r".\emd_files\EDS_net_20260107_1508_SI_265_kx_HAADF.emd"
    # file_path = r".\emd_files\EDS_colormix_20260119_1108_SI_135_kx_HAADF.emd"
    # file_path = r".\emd_files\TEM_Single_20250619_630kx_Ceta_0002.emd"
    # file_path = r".\emd_files\TEM_Series_20250619_630kx_Ceta_0001.emd"
    # file_path = r".\emd_files\STEM_Single_20250620_STEM_380kx_HAADF.emd"
    # file_path = r".\emd_files\DPC_20250619_DPC_2.15Mx_DF_0006.emd"
    # file_path = r".\emd_files\DPC_20250619_DPC_2.15Mx_DF_0015.emd"
    # file_path = r".\emd_files\Insitu-dianhuaxue Camera 1.05 Mx Ceta 1326.emd"
    # file_path = r".\emd_files\STEM_Single_Series_4100_x_HAADF.emd"
    # file_path = r".\emd_files\STEM_Multi_Series_380_kx_HAADF-DF-DF-I-DF-O-BF.emd"
    # file_path = r".\emd_files\SAD_Single_Camera_260mm_Ceta_20250603_1312.emd"
    # file_path = r".\emd_files\SAD_Series_20250620_660mm_Ceta.emd"
    
    # directory = Path(r"F:\20260303-lyx")
    # for file_path in directory.glob("*.emd"):
    # for file_path in list(directory.glob("*.emd"))[:1]:
        # analyzer = VeloxFileAnalyzer(file_path)
        # analyzer.export()
    # exit()
    
    # 创建分析器
    analyzer = VeloxFileAnalyzer(file_path)
    analyzer.export()
    # 访问 parameters
    # print("基本参数：")
    # print(json.dumps(analyzer.parameters, indent=4))
    
    # 根据检测到的特征类型访问数据
    if hasattr(analyzer, 'si_feature_path'):
        # EDS mapping 数据
        print("检测到 EDS mapping 特征")
        print(f"可用 mapping 数据: {list(analyzer.mapping_data.keys())}")
        print(f"谱图数据: {list(analyzer.spectra_data.keys())}")
        # analyzer.display()
        # 如果需要定量结果
        # quantification_result = html_table_to_csv(analyzer.experiment_log)
        # print(quantification_result)
    
    if hasattr(analyzer, 'color_mix_profile_feature_path'):
        # 颜色混合和线剖面数据
        print("检测到颜色混合和线剖面特征")
        print(f"线剖面数据: {list(analyzer.line_profile_data.keys())}")
        # fig, ax = draw_line_annotation_on_image(analyzer.color_mix_image, line_info=analyzer.line_position, pixel_size=analyzer.parameters['pixelsize'], pixel_unit=analyzer.parameters['pixelunit'])
        # fig, ax = draw_line_annotation_on_image(analyzer.color_mix_image, pixel_size=analyzer.parameters['pixelsize'], pixel_unit=analyzer.parameters['pixelunit'])
        # plt.show()
        # fig = draw_line_profiles(analyzer.line_profile_data, pixel_size=analyzer.parameters['pixelsize'], pixel_unit=analyzer.parameters['pixelunit'])
        # plt.show()
    
    if hasattr(analyzer, 'camera_feature_path'):
        # TEM 图像数据
        """
        tem_data：图像数据 np.array
        tem_metadata：原始的metadata
        parameters['Ceta']：提取的关键metadata，包含pixelsize, pixelunit, display_index, display_range, gamma
        """
        print("检测到 TEM 图像特征")
        print(f"图像形状: {analyzer.tem_data.shape}")
        for key, value in analyzer.parameters.items():
            if key == 'Ceta':
                print(f"图像形状（{key}）: {analyzer.tem_data.shape}")
                # fig, axe = display_image_with_scale(image = analyzer.tem_data, pixel_size = value['pixelsize'], pixel_unit = value['pixelunit'], title = value['image_name'], cmap = 'gray', display_range = value['display_range'], gamma = value['gamma'], display_index = value['display_index'])
                # plt.show()
    
    if hasattr(analyzer, 'stem_feature_path'):
        # STEM 图像数据
        print("检测到 STEM 图像特征")
        for key, value in analyzer.parameters.items():
            if key in analyzer.stem_data:
                print(f"图像形状（{key}）: {analyzer.stem_data[key].shape}")
                # fig, axe = display_image_with_scale(image = analyzer.stem_data[key], pixel_size = value['pixelsize'], pixel_unit = value['pixelunit'], title = value['image_name'], cmap = 'gray', display_range = value['display_range'], gamma = value['gamma'], display_index = value['display_index'])
                # plt.show()
    
    if hasattr(analyzer, 'dcfi_feature_path'):
        # DCFI 数据
        """
        parameters['DCFI'] = {
            'image_name': f"{unquote(dcfi_display['display']['label'])} of {Path(self.file_path).stem}",
            'pixelsize': self._get_pixel_size(image_metadata)[0],
            'pixelunit': self._get_pixel_size(image_metadata)[1],
            'display_index': int(dcfi_display['seriesIndex']),
            'display_range': [
                float(item) for item in dcfi_display['displayLevelsRange'].values()
            ],
            'gamma': float(dcfi_display['gamma'])
        }
        """
        print("检测到 DCFI 特征")
        print(f"DCFI 数据形状: {analyzer.dcfi_data.shape}")
        for key, value in analyzer.parameters.items():
            if key == 'DCFI':
                print(f"图像形状（{key}）: {analyzer.dcfi_data.shape}")
                # fig, axe = display_image_with_scale(image = analyzer.dcfi_data, pixel_size = value['pixelsize'], pixel_unit = value['pixelunit'], title = value['image_name'], cmap = 'gray', display_range = value['display_range'], gamma = value['gamma'], display_index = value['display_index'])
                # plt.show()
    
    if hasattr(analyzer, 'dpc_feature_path'):
        # DPC 数据
        print("检测到 DPC 特征")
        print(f"DPC 图像: {list(analyzer.dpc_data.keys())}")
        for key, value in analyzer.parameters.items():
            if key in analyzer.dpc_data:
                print(f"图像形状（{key}）: {analyzer.dpc_data[key].shape}")
                # fig, axe = display_image_with_scale(image = analyzer.dpc_data[key], pixel_size = value['pixelsize'], pixel_unit = value['pixelunit'], title = value['image_name'], cmap = 'gray', display_range = value['display_range'], gamma = value['gamma'], display_index = value['display_index'])
                # plt.show()
    
    if hasattr(analyzer, 'crop_feature_path'):
        # 裁剪图像
        print("检测到裁剪图像特征")
        print(f"裁剪图像形状: {analyzer.crop_data.shape}")
        # key = 'crop'
        # value = analyzer.parameters[key]
        # fig, axe = display_image_with_scale(image = analyzer.crop_data, pixel_size = value['pixelsize'], pixel_unit = value['pixelunit'], title = value['image_name'], cmap = 'gray', display_range = value['display_range'], gamma = value['gamma'], display_index = value['display_index'])
        # plt.show()
    
    if hasattr(analyzer, 'image_filter_feature_path'):
        # 滤波图像
        print("检测到滤波图像特征")
        print(f"滤波类型: {analyzer.parameters['filter']['filter_type']}")
        # key = 'filter'
        # value = analyzer.parameters[key]
        # fig, axe = display_image_with_scale(image = analyzer.crop_data, pixel_size = value['pixelsize'], pixel_unit = value['pixelunit'], title = value['image_name'], cmap = 'gray', display_range = value['display_range'], gamma = value['gamma'], display_index = value['display_index'])
        # plt.show()
        
        