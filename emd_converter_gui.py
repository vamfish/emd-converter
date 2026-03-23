"""
EMD 文件批量转换工具 - GUI 版本
基于 velox_file_analyzer2.py 的功能封装
"""

import os
import sys
import json
import threading
from pathlib import Path
from datetime import datetime
from tkinter import (
    Tk, Frame, Label, Button, Entry, Checkbutton, 
    Listbox, Scrollbar, StringVar, BooleanVar, 
    filedialog, messagebox, ttk, IntVar, DoubleVar,
    Menu
)
import tkinter as tk

import numpy as np

# 设置 Matplotlib 使用非交互式后端（必须在导入 velox_file_analyzer2 之前）
import matplotlib
matplotlib.use('Agg')

# 导入核心功能
from velox_file_analyzer2 import VeloxFileAnalyzer

# 配置文件路径
CONFIG_FILE = Path(__file__).parent / "gui_config.json"


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


class EMDConverterGUI:
    """EMD 转换器图形界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("EMD 文件批量转换工具")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # 数据存储
        self.file_list = []
        self.output_dir = StringVar()
        self.is_processing = False
        
        # 导出选项变量
        self.export_options = {
            'dm5': BooleanVar(value=True),
            'tiff': BooleanVar(value=True),
            'png': BooleanVar(value=True),
            'csv': BooleanVar(value=False),
        }
        
        # EDS Mapping 特有选项
        self.eds_options = {
            'export_colormix': BooleanVar(value=True),
            'export_elements': BooleanVar(value=True),
            'export_haadf': BooleanVar(value=True),
        }
        
        # Color Mix 特有选项
        self.colormix_options = {
            'all_elements': BooleanVar(value=True),
            'with_annotation': BooleanVar(value=True),
        }
        
        # 加载保存的配置
        self._load_config()
        
        # 创建界面
        self._create_widgets()
        self._layout_widgets()
        self._create_menu()
        
        # 重定向标准输出到日志区域
        self._redirect_stdout()
        
        # 窗口关闭时保存配置
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _create_widgets(self):
        """创建所有界面组件"""
        
        # === 文件选择区域 ===
        self.file_frame = Frame(self.root)
        
        self.file_label = Label(self.file_frame, text="EMD 文件:", font=('Microsoft YaHei', 10))
        
        self.file_listbox = Listbox(self.file_frame, selectmode='extended', height=8)
        self.file_scrollbar = Scrollbar(self.file_frame, orient='vertical', command=self.file_listbox.yview)
        self.file_listbox.config(yscrollcommand=self.file_scrollbar.set)
        
        self.btn_add_files = Button(self.file_frame, text="添加文件", command=self._add_files)
        self.btn_add_folder = Button(self.file_frame, text="添加文件夹", command=self._add_folder)
        self.btn_remove = Button(self.file_frame, text="移除选中", command=self._remove_selected)
        self.btn_clear = Button(self.file_frame, text="清空列表", command=self._clear_files)
        
        # === 输出目录区域 ===
        self.output_frame = Frame(self.root)
        
        self.output_label = Label(self.output_frame, text="输出目录:", font=('Microsoft YaHei', 10))
        self.output_entry = Entry(self.output_frame, textvariable=self.output_dir, width=60)
        self.btn_browse_output = Button(self.output_frame, text="浏览...", command=self._browse_output)
        self.btn_same_as_input = Button(self.output_frame, text="与输入相同", command=self._set_same_as_input)
        
        # === 导出选项区域 ===
        self.options_frame = Frame(self.root, relief='groove', bd=2)
        self.options_label = Label(self.options_frame, text="导出选项", font=('Microsoft YaHei', 12, 'bold'))
        
        # 基础格式选项
        self.format_frame = Frame(self.options_frame)
        self.cb_dm5 = Checkbutton(self.format_frame, text="DM5 格式", variable=self.export_options['dm5'])
        self.cb_tiff = Checkbutton(self.format_frame, text="16-bit TIFF", variable=self.export_options['tiff'])
        self.cb_png = Checkbutton(self.format_frame, text="PNG 图像(带比例尺)", variable=self.export_options['png'])
        self.cb_csv = Checkbutton(self.format_frame, text="CSV 数据", variable=self.export_options['csv'])
        
        # EDS Mapping 选项
        self.eds_frame = Frame(self.options_frame)
        self.eds_label = Label(self.eds_frame, text="EDS Mapping:", font=('Microsoft YaHei', 10, 'bold'))
        self.cb_colormix = Checkbutton(self.eds_frame, text="导出 Color Mix", variable=self.eds_options['export_colormix'])
        self.cb_elements = Checkbutton(self.eds_frame, text="导出各元素分布图", variable=self.eds_options['export_elements'])
        self.cb_haadf = Checkbutton(self.eds_frame, text="导出 HAADF", variable=self.eds_options['export_haadf'])
        
        # Color Mix 选项
        self.colormix_frame = Frame(self.options_frame)
        self.colormix_label = Label(self.colormix_frame, text="Color Mix:", font=('Microsoft YaHei', 10, 'bold'))
        self.cb_all_elements = Checkbutton(self.colormix_frame, text="包含所有元素", variable=self.colormix_options['all_elements'])
        self.cb_with_annotation = Checkbutton(self.colormix_frame, text="包含线扫描标注", variable=self.colormix_options['with_annotation'])
        
        # === 进度区域 ===
        self.progress_frame = Frame(self.root)
        self.progress_label = Label(self.progress_frame, text="准备就绪", font=('Microsoft YaHei', 10))
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate', length=400)
        self.progress_var = DoubleVar()
        self.progress_bar.config(variable=self.progress_var)
        
        # === 按钮区域 ===
        self.button_frame = Frame(self.root)
        self.btn_start = Button(self.button_frame, text="开始转换", command=self._start_conversion, 
                               bg='#4CAF50', fg='white', font=('Microsoft YaHei', 11, 'bold'), padx=20, pady=5)
        self.btn_stop = Button(self.button_frame, text="停止", command=self._stop_conversion,
                              bg='#f44336', fg='white', font=('Microsoft YaHei', 11), padx=20, pady=5, state='disabled')
        
        # === 日志区域 ===
        self.log_frame = Frame(self.root)
        self.log_label = Label(self.log_frame, text="处理日志:", font=('Microsoft YaHei', 10))
        self.log_text = tk.Text(self.log_frame, height=12, wrap='word', state='disabled')
        self.log_scrollbar = Scrollbar(self.log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.config(yscrollcommand=self.log_scrollbar.set)
        
    def _layout_widgets(self):
        """布局所有组件"""
        
        # 文件选择区域
        self.file_frame.pack(fill='x', padx=10, pady=5)
        self.file_label.pack(anchor='w')
        self.file_listbox.pack(side='left', fill='both', expand=True)
        self.file_scrollbar.pack(side='left', fill='y')
        
        btn_frame = Frame(self.file_frame)
        btn_frame.pack(side='left', padx=5)
        self.btn_add_files.pack(fill='x', pady=2)
        self.btn_add_folder.pack(fill='x', pady=2)
        self.btn_remove.pack(fill='x', pady=2)
        self.btn_clear.pack(fill='x', pady=2)
        
        # 输出目录区域
        self.output_frame.pack(fill='x', padx=10, pady=5)
        self.output_label.pack(side='left')
        self.output_entry.pack(side='left', padx=5, fill='x', expand=True)
        self.btn_browse_output.pack(side='left', padx=2)
        self.btn_same_as_input.pack(side='left', padx=2)
        
        # 导出选项区域
        self.options_frame.pack(fill='x', padx=10, pady=5)
        self.options_label.pack(anchor='w', padx=5, pady=5)
        
        # 格式选项
        self.format_frame.pack(fill='x', padx=10, pady=2)
        self.cb_dm5.pack(side='left', padx=10)
        self.cb_tiff.pack(side='left', padx=10)
        self.cb_png.pack(side='left', padx=10)
        self.cb_csv.pack(side='left', padx=10)
        
        # EDS 选项
        self.eds_frame.pack(fill='x', padx=10, pady=2)
        self.eds_label.pack(side='left', padx=5)
        self.cb_colormix.pack(side='left', padx=10)
        self.cb_elements.pack(side='left', padx=10)
        self.cb_haadf.pack(side='left', padx=10)
        
        # Color Mix 选项
        self.colormix_frame.pack(fill='x', padx=10, pady=2)
        self.colormix_label.pack(side='left', padx=5)
        self.cb_all_elements.pack(side='left', padx=10)
        self.cb_with_annotation.pack(side='left', padx=10)
        
        # 进度区域
        self.progress_frame.pack(fill='x', padx=10, pady=5)
        self.progress_label.pack(anchor='w')
        self.progress_bar.pack(fill='x', pady=2)
        
        # 按钮区域
        self.button_frame.pack(pady=10)
        self.btn_start.pack(side='left', padx=10)
        self.btn_stop.pack(side='left', padx=10)
        
        # 日志区域
        self.log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.log_label.pack(anchor='w')
        self.log_text.pack(side='left', fill='both', expand=True)
        self.log_scrollbar.pack(side='left', fill='y')
        
    def _create_menu(self):
        """创建菜单栏"""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="添加文件...", command=self._add_files)
        file_menu.add_command(label="添加文件夹...", command=self._add_folder)
        file_menu.add_separator()
        file_menu.add_command(label="保存配置", command=self._save_config)
        file_menu.add_command(label="加载配置", command=self._load_config_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_closing)
        
        # 帮助菜单
        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self._show_help)
        help_menu.add_command(label="关于", command=self._show_about)
        
    def _redirect_stdout(self):
        """重定向标准输出到日志区域"""
        class StdoutRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
                
            def write(self, text):
                self.text_widget.config(state='normal')
                self.text_widget.insert('end', text)
                self.text_widget.see('end')
                self.text_widget.config(state='disabled')
                
            def flush(self):
                pass
                
        sys.stdout = StdoutRedirector(self.log_text)
        
    def _load_config(self):
        """加载配置文件"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 加载导出选项
                for key, value in config.get('export_options', {}).items():
                    if key in self.export_options:
                        self.export_options[key].set(value)
                        
                for key, value in config.get('eds_options', {}).items():
                    if key in self.eds_options:
                        self.eds_options[key].set(value)
                        
                for key, value in config.get('colormix_options', {}).items():
                    if key in self.colormix_options:
                        self.colormix_options[key].set(value)
                        
                # 加载输出目录
                output_dir = config.get('output_dir', '')
                if output_dir and Path(output_dir).exists():
                    self.output_dir.set(output_dir)
                    
            except Exception as e:
                print(f"加载配置失败: {e}")
                
    def _save_config(self):
        """保存配置文件"""
        try:
            config = {
                'export_options': {k: v.get() for k, v in self.export_options.items()},
                'eds_options': {k: v.get() for k, v in self.eds_options.items()},
                'colormix_options': {k: v.get() for k, v in self.colormix_options.items()},
                'output_dir': self.output_dir.get(),
            }
            
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
            self._log("配置已保存")
            messagebox.showinfo("成功", "配置已保存")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {e}")
            
    def _load_config_dialog(self):
        """从对话框加载配置"""
        filename = filedialog.askopenfilename(
            title="选择配置文件",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                for key, value in config.get('export_options', {}).items():
                    if key in self.export_options:
                        self.export_options[key].set(value)
                        
                for key, value in config.get('eds_options', {}).items():
                    if key in self.eds_options:
                        self.eds_options[key].set(value)
                        
                for key, value in config.get('colormix_options', {}).items():
                    if key in self.colormix_options:
                        self.colormix_options[key].set(value)
                        
                self._log(f"配置已加载: {filename}")
                messagebox.showinfo("成功", "配置已加载")
                
            except Exception as e:
                messagebox.showerror("错误", f"加载配置失败: {e}")
                
    def _on_closing(self):
        """窗口关闭时的处理"""
        if self.is_processing:
            if not messagebox.askokcancel("确认", "正在处理中，确定要退出吗?"):
                return
            self.is_processing = False
        
        # 保存配置
        self._save_config()
        
        # 清理 Matplotlib 资源，避免 RuntimeError
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
        # 销毁窗口
        self.root.destroy()
            
    def _show_help(self):
        """显示帮助信息"""
        help_text = """
EMD 文件批量转换工具使用说明

1. 添加文件：
   - 点击"添加文件"选择单个或多个 EMD 文件
   - 或点击"添加文件夹"添加整个文件夹中的 EMD 文件

2. 设置输出目录：
   - 点击"浏览..."选择输出位置
   - 或点击"与输入相同"使用输入文件所在目录

3. 选择导出格式：
   - DM5: Digital Micrograph 原生格式
   - TIFF: 16-bit 格式，ImageJ 兼容
   - PNG: 带比例尺的图像，适合论文
   - CSV: 定量数据和谱图数据

4. 点击"开始转换"按钮开始处理

注意：
- 确保 EMD 文件没有被其他软件打开
- 大批量处理可能需要较长时间
- 输出文件将保存在 "输出目录/custom_export/文件名/" 下
        """
        messagebox.showinfo("使用说明", help_text)
        
    def _show_about(self):
        """显示关于信息"""
        about_text = """
EMD 文件批量转换工具 v1.0

基于 velox_file_analyzer2.py 开发
用于将 Velox EMD 文件转换为各种格式

支持的数据类型：
- EDS Mapping
- TEM/STEM 图像
- DPC 图像
- Color Mix + Line Profile
- DCFI 图像

作者: 基于 velox_file_analyzer2.py 修改
        """
        messagebox.showinfo("关于", about_text)
        
    def _add_files(self):
        """添加单个文件"""
        files = filedialog.askopenfilenames(
            title="选择 EMD 文件",
            filetypes=[("EMD 文件", "*.emd"), ("所有文件", "*.*")]
        )
        for f in files:
            if f not in self.file_list:
                self.file_list.append(f)
                self.file_listbox.insert('end', Path(f).name)
                
    def _add_folder(self):
        """添加整个文件夹"""
        folder = filedialog.askdirectory(title="选择包含 EMD 文件的文件夹")
        if folder:
            emd_files = list(Path(folder).rglob("*.emd"))
            for f in emd_files:
                f_str = str(f)
                if f_str not in self.file_list:
                    self.file_list.append(f_str)
                    self.file_listbox.insert('end', f.name)
                    
    def _remove_selected(self):
        """移除选中的文件"""
        selected = self.file_listbox.curselection()
        for idx in reversed(selected):
            self.file_list.pop(idx)
            self.file_listbox.delete(idx)
            
    def _clear_files(self):
        """清空文件列表"""
        self.file_list.clear()
        self.file_listbox.delete(0, 'end')
        
    def _browse_output(self):
        """浏览输出目录"""
        folder = filedialog.askdirectory(title="选择输出目录")
        if folder:
            self.output_dir.set(folder)
            
    def _set_same_as_input(self):
        """设置输出目录为输入文件所在目录"""
        if self.file_list:
            first_file = Path(self.file_list[0])
            self.output_dir.set(str(first_file.parent))
        else:
            messagebox.showwarning("警告", "请先添加 EMD 文件")
            
    def _log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def _start_conversion(self):
        """开始转换"""
        if not self.file_list:
            messagebox.showwarning("警告", "请先添加 EMD 文件")
            return
            
        if not self.output_dir.get():
            # 默认使用第一个文件所在目录
            first_file = Path(self.file_list[0])
            self.output_dir.set(str(first_file.parent / "custom_export"))
            
        # 检查至少选择了一种导出格式
        if not any(var.get() for var in self.export_options.values()):
            messagebox.showwarning("警告", "请至少选择一种导出格式")
            return
            
        # 在后台线程中运行转换
        self.is_processing = True
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.progress_var.set(0)
        
        thread = threading.Thread(target=self._process_files)
        thread.daemon = True
        thread.start()
        
    def _stop_conversion(self):
        """停止转换"""
        self.is_processing = False
        self._log("正在停止...")
        
    def _process_files(self):
        """处理所有文件"""
        total = len(self.file_list)
        output_base = Path(self.output_dir.get())
        
        for idx, file_path in enumerate(self.file_list, 1):
            if not self.is_processing:
                break
                
            try:
                self.root.after(0, lambda i=idx, t=total: self._update_progress(i, t, f"正在处理: {Path(file_path).name}"))
                self._log(f"\n{'='*50}")
                self._log(f"开始处理 ({idx}/{total}): {file_path}")
                
                # 分析文件
                analyzer = VeloxFileAnalyzer(file_path)
                
                # 确定输出目录
                file_output_dir = output_base / Path(file_path).stem
                file_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 根据文件类型执行导出
                self._export_by_type(analyzer, file_output_dir)
                
                self._log(f"完成: {Path(file_path).name}")
                
            except Exception as e:
                self._log(f"错误: 处理 {Path(file_path).name} 时出错: {str(e)}")
                import traceback
                self._log(traceback.format_exc())
                
        self.root.after(0, self._conversion_finished)
        
    def _update_progress(self, current, total, message):
        """更新进度"""
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.progress_label.config(text=f"{message} ({current}/{total})")
        
    def _conversion_finished(self):
        """转换完成回调"""
        self.is_processing = False
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        self.progress_var.set(100)
        self.progress_label.config(text="处理完成")
        self._log(f"\n{'='*50}")
        self._log("所有文件处理完成!")
        messagebox.showinfo("完成", "EMD 文件转换完成!")
        
    def _export_by_type(self, analyzer, output_dir):
        """根据文件类型执行导出"""
        filename_stem = Path(analyzer.file_path).stem
        
        # 获取导出选项
        export_dm5 = self.export_options['dm5'].get()
        export_tiff = self.export_options['tiff'].get()
        export_png = self.export_options['png'].get()
        export_csv = self.export_options['csv'].get()
        
        # EDS Mapping 类型
        if hasattr(analyzer, 'si_feature_path'):
            self._log("检测到 EDS Mapping 数据")
            self._export_eds_mapping(analyzer, output_dir, filename_stem)
            
        # 积分谱图类型
        if hasattr(analyzer, 'integrated_spectra_feature_path'):
            self._log("检测到 EDS 积分谱图")
            if export_csv:
                self._export_spectra(analyzer, output_dir, filename_stem)
                
        # Color Mix + Line Profile
        if hasattr(analyzer, 'color_mix_profile_feature_path'):
            self._log("检测到 Color Mix + Line Profile")
            self._export_colormix_lineprofile(analyzer, output_dir, filename_stem)
            
        # TEM 图像
        if hasattr(analyzer, 'camera_feature_path'):
            self._log("检测到 TEM 图像")
            self._export_tem(analyzer, output_dir, filename_stem)
            
        # STEM 图像
        if hasattr(analyzer, 'stem_feature_path'):
            self._log("检测到 STEM 图像")
            self._export_stem(analyzer, output_dir, filename_stem)
            
        # DPC 图像
        if hasattr(analyzer, 'dpc_feature_path'):
            self._log("检测到 DPC 图像")
            self._export_dpc(analyzer, output_dir, filename_stem)
            
        # DCFI 图像
        if hasattr(analyzer, 'dcfi_feature_path'):
            self._log("检测到 DCFI 图像")
            self._export_dcfi(analyzer, output_dir, filename_stem)
            
        # 裁剪图像
        if hasattr(analyzer, 'crop_feature_path'):
            self._log("检测到裁剪图像")
            self._export_crop(analyzer, output_dir, filename_stem)
            
        # 滤波图像
        if hasattr(analyzer, 'image_filter_feature_path'):
            self._log("检测到滤波图像")
            self._export_filtered(analyzer, output_dir, filename_stem)
            
    def _export_eds_mapping(self, analyzer, output_dir, filename_stem):
        """导出 EDS Mapping 数据"""
        from velox_file_analyzer2 import (
            dm5_writer, save_as_16bit_tiff, save_image_as_png,
            LinearSegmentedColormap
        )
        
        quantification_mode = analyzer.parameters.get('quantification_mode', 'Unknown')
        
        # 导出定量结果
        if self.export_options['csv'].get():
            try:
                from velox_file_analyzer2 import html_table_to_csv
                output_path = output_dir / f"{filename_stem}-Quantification-{quantification_mode}.csv"
                html_table_to_csv(analyzer.experiment_log, output_path=str(output_path))
                self._log(f"  ✓ 定量结果: {output_path.name}")
            except Exception as e:
                self._log(f"  ✗ 导出定量结果失败: {e}")
                
        # 导出 Color Mix
        if self.eds_options['export_colormix'].get() and hasattr(analyzer, 'color_mix_image'):
            from velox_file_analyzer2 import save_color_mix_image
            try:
                output_path = output_dir / f"{filename_stem}-Colormix.png"
                save_color_mix_image(
                    image=analyzer.color_mix_image,
                    output_path=output_path,
                    pixel_size=analyzer.parameters['pixelsize'],
                    pixel_unit=analyzer.parameters['pixelunit']
                )
                self._log(f"  ✓ Color Mix: {output_path.name}")
            except Exception as e:
                self._log(f"  ✗ 导出 Color Mix 失败: {e}")
                
        # 导出各元素分布图和 HAADF
        if self.eds_options['export_elements'].get() or self.eds_options['export_haadf'].get():
            for key, value in analyzer.mapping_data.items():
                # 判断是否 HAADF
                is_haadf = 'HAADF' in key or 'BF' in key or 'DF' in key
                
                if is_haadf and not self.eds_options['export_haadf'].get():
                    continue
                if not is_haadf and not self.eds_options['export_elements'].get():
                    continue
                    
                filename = f"{filename_stem}-{key}-{quantification_mode}"
                data = value['data'][:, :, value['frame_index']]
                output_path = output_dir / filename
                
                try:
                    # DM5
                    if self.export_options['dm5'].get():
                        dm5_writer(add_suffix_safe(output_path, '.dm5'), value, analyzer.parameters)
                        self._log(f"  ✓ DM5: {filename}.dm5")
                        
                    # TIFF
                    if self.export_options['tiff'].get():
                        imagej_metadata = {
                            'ImageJ': '1.54g',
                            'unit': analyzer.parameters['pixelunit'].replace('μ', 'u'),
                            'spacing': analyzer.parameters['pixelsize']
                        }
                        if data.ndim == 3:
                            imagej_metadata['slices'] = data.shape[-1]
                        save_as_16bit_tiff(data, output_path, metadata=imagej_metadata)
                        self._log(f"  ✓ TIFF: {filename}.tif")
                        
                    # PNG
                    if self.export_options['png'].get():
                        color_rgb = (value['color']['red'], value['color']['green'], value['color']['blue'])
                        cmap = LinearSegmentedColormap.from_list(
                            'custom_cmap',
                            colors=[(0, 0, 0), color_rgb],
                            N=256
                        )
                        save_image_as_png(
                            image=data,
                            output_path=add_suffix_safe(output_path, '.png'),
                            pixel_size=analyzer.parameters['pixelsize'],
                            pixel_unit=analyzer.parameters['pixelunit'],
                            cmap=cmap,
                            display_range=value['display_range'],
                            gamma=value['gamma'],
                            display_index=value['frame_index'],
                            add_scalebar=True
                        )
                        self._log(f"  ✓ PNG: {filename}.png")
                        
                except Exception as e:
                    self._log(f"  ✗ 导出 {key} 失败: {e}")
                    
    def _export_spectra(self, analyzer, output_dir, filename_stem):
        """导出 EDS 谱图"""
        from velox_file_analyzer2 import export_eds_spectrum
        try:
            filename = f"{filename_stem}-Spectra.csv"
            output_path = output_dir / filename
            export_eds_spectrum(
                output_path=output_path,
                intensity=analyzer.spectra_data['total'],
                offset=analyzer.parameters.get('OffsetEnergy', -250),
                channels=4096,
                dispension=5
            )
            self._log(f"  ✓ 谱图 CSV: {filename}")
        except Exception as e:
            self._log(f"  ✗ 导出谱图失败: {e}")
            
    def _export_colormix_lineprofile(self, analyzer, output_dir, filename_stem):
        """导出 Color Mix 和 Line Profile"""
        from velox_file_analyzer2 import (
            save_color_mix_image, draw_line_profiles,
            export_line_profile_as_csv
        )
        
        quantification_mode = analyzer.parameters.get('quantification_mode', 'Unknown')
        
        # 导出带标注的 Color Mix
        if self.eds_options['export_colormix'].get() and hasattr(analyzer, 'color_mix_image'):
            try:
                filename = f"{filename_stem}-Colormix-LineAnnotation.png"
                output_path = output_dir / filename
                
                if self.colormix_options['with_annotation'].get() and hasattr(analyzer, 'line_position'):
                    save_color_mix_image(
                        analyzer.color_mix_image,
                        output_path=output_path,
                        line_info=analyzer.line_position,
                        pixel_size=analyzer.parameters['pixelsize'],
                        pixel_unit=analyzer.parameters['pixelunit']
                    )
                else:
                    save_color_mix_image(
                        analyzer.color_mix_image,
                        output_path=output_path,
                        pixel_size=analyzer.parameters['pixelsize'],
                        pixel_unit=analyzer.parameters['pixelunit']
                    )
                self._log(f"  ✓ Color Mix (带标注): {filename}")
            except Exception as e:
                self._log(f"  ✗ 导出 Color Mix 失败: {e}")
                
        # 导出 Line Profile 图像
        if self.export_options['png'].get() and hasattr(analyzer, 'line_profile_data'):
            try:
                filename = f"{filename_stem}-LineProfile.png"
                output_path = output_dir / filename
                draw_line_profiles(
                    analyzer.line_profile_data,
                    output_path=output_path,
                    pixel_size=analyzer.parameters['pixelsize'],
                    pixel_unit=analyzer.parameters['pixelunit'],
                    line_length_px=analyzer.line_position.get('length') if hasattr(analyzer, 'line_position') else None
                )
                self._log(f"  ✓ Line Profile 图像: {filename}")
            except Exception as e:
                self._log(f"  ✗ 导出 Line Profile 图像失败: {e}")
                
        # 导出 Line Profile CSV
        if self.export_options['csv'].get() and hasattr(analyzer, 'line_profile_data'):
            try:
                filename = f"{filename_stem}-LineProfile.csv"
                output_path = output_dir / filename
                export_line_profile_as_csv(
                    output_path,
                    line_length_pixel=analyzer.line_position.get('length') if hasattr(analyzer, 'line_position') else None,
                    line_profile=analyzer.line_profile_data,
                    pixelsize=analyzer.parameters['pixelsize'],
                    pixelunit=analyzer.parameters['pixelunit'],
                    quantification_mode=quantification_mode
                )
                self._log(f"  ✓ Line Profile CSV: {filename}")
            except Exception as e:
                self._log(f"  ✗ 导出 Line Profile CSV 失败: {e}")
                
    def _export_tem(self, analyzer, output_dir, filename_stem):
        """导出 TEM 图像"""
        self._export_generic_image(
            analyzer, output_dir, filename_stem,
            'Ceta', analyzer.tem_data, analyzer.tem_metadata
        )
        
    def _export_stem(self, analyzer, output_dir, filename_stem):
        """导出 STEM 图像"""
        for key in analyzer.stem_data.keys():
            params = analyzer.parameters.get(key, {})
            self._export_generic_image(
                analyzer, output_dir, filename_stem,
                key, analyzer.stem_data[key], analyzer.stem_metadata.get(key, {}),
                suffix=key
            )
            
    def _export_dpc(self, analyzer, output_dir, filename_stem):
        """导出 DPC 图像"""
        for key in analyzer.dpc_data.keys():
            params = analyzer.parameters.get(key, {})
            self._export_generic_image(
                analyzer, output_dir, filename_stem,
                key, analyzer.dpc_data[key], analyzer.dpc_metadata.get(key, {}),
                suffix=key
            )
            
    def _export_dcfi(self, analyzer, output_dir, filename_stem):
        """导出 DCFI 图像"""
        params = analyzer.parameters.get('DCFI', {})
        image_name = params.get('image_name', f"{filename_stem}-DCFI")
        self._export_generic_image(
            analyzer, output_dir, filename_stem,
            'DCFI', analyzer.dcfi_data, {},
            custom_name=image_name
        )
        
    def _export_crop(self, analyzer, output_dir, filename_stem):
        """导出裁剪图像"""
        params = analyzer.parameters.get('crop', {})
        image_name = params.get('image_name', f"{filename_stem}-Crop")
        self._export_generic_image(
            analyzer, output_dir, filename_stem,
            'crop', analyzer.crop_data, analyzer.crop_metadata,
            custom_name=image_name
        )
        
    def _export_filtered(self, analyzer, output_dir, filename_stem):
        """导出滤波图像"""
        params = analyzer.parameters.get('filter', {})
        image_name = params.get('image_name', f"{filename_stem}-Filtered")
        self._export_generic_image(
            analyzer, output_dir, filename_stem,
            'filter', analyzer.filter_data, analyzer.filter_metadata,
            custom_name=image_name
        )
        
    def _export_generic_image(self, analyzer, output_dir, filename_stem, 
                              param_key, data, metadata, suffix=None, custom_name=None):
        """通用图像导出方法"""
        from velox_file_analyzer2 import dm5_writer, save_as_16bit_tiff, save_image_as_png
        
        params = analyzer.parameters.get(param_key, {}) if isinstance(analyzer.parameters.get(param_key), dict) else {}
        
        if custom_name:
            filename = custom_name
        elif suffix:
            filename = f"{filename_stem}-{suffix}"
        else:
            filename = filename_stem
            
        output_path = output_dir / filename
        
        try:
            signal = {
                'data': data,
                'metadata': metadata,
                'color': {'blue': 1, 'green': 1, 'red': 1},
                'display_range': params.get('display_range', [0, 1]),
                'gamma': params.get('gamma', 1.0)
            }
            
            # DM5
            if self.export_options['dm5'].get():
                dm5_data = data
                if dm5_data.ndim == 2:
                    dm5_data = dm5_data[..., np.newaxis]
                signal['data'] = dm5_data
                dm5_writer(add_suffix_safe(output_path, '.dm5'), signal, params if params else analyzer.parameters)
                self._log(f"  ✓ DM5: {filename}.dm5")
                
            # TIFF
            if self.export_options['tiff'].get():
                imagej_metadata = {
                    'ImageJ': '1.54g',
                    'unit': params.get('pixelunit', 'nm').replace('μ', 'u') if params else 'nm',
                    'spacing': params.get('pixelsize', 1.0) if params else 1.0
                }
                if data.ndim == 3:
                    imagej_metadata['slices'] = data.shape[-1]
                save_as_16bit_tiff(data, output_path, metadata=imagej_metadata)
                self._log(f"  ✓ TIFF: {filename}.tif")
                
            # PNG
            if self.export_options['png'].get():
                if data.ndim == 2:
                    save_image_as_png(
                        image=data,
                        output_path=add_suffix_safe(output_path, '.png'),
                        pixel_size=params.get('pixelsize', 1.0) if params else 1.0,
                        pixel_unit=params.get('pixelunit', 'nm') if params else 'nm',
                        display_range=params.get('display_range'),
                        gamma=params.get('gamma', 1.0),
                        add_scalebar=True
                    )
                elif data.ndim == 3:
                    save_image_as_png(
                        image=data,
                        output_path=add_suffix_safe(output_path, '.png'),
                        pixel_size=params.get('pixelsize', 1.0) if params else 1.0,
                        pixel_unit=params.get('pixelunit', 'nm') if params else 'nm',
                        display_range=params.get('display_range'),
                        gamma=params.get('gamma', 1.0),
                        display_index=params.get('display_index', 0),
                        add_scalebar=True
                    )
                self._log(f"  ✓ PNG: {filename}.png")
                
        except Exception as e:
            self._log(f"  ✗ 导出 {filename} 失败: {e}")


def main():
    """主函数"""
    root = Tk()
    app = EMDConverterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
