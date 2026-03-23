"""
EMD Converter GUI Launcher
启动 EMD 文件转换器图形界面

用法:
    python launch_gui.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """启动 GUI"""
    project_dir = Path(__file__).parent.absolute()
    gui_script = project_dir / "emd_converter_gui.py"
    
    # 尝试使用不同的 Python 解释器
    python_paths = [
        sys.executable,  # 当前 Python
        "python",
        "python3",
        "pythonw",
    ]
    
    for python_exe in python_paths:
        try:
            result = subprocess.run(
                [python_exe, str(gui_script)],
                check=False,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return
        except FileNotFoundError:
            continue
    
    print("错误: 无法启动 GUI")
    print("请确保已安装 Python 和所需依赖:")
    print("  pip install -r requirements.txt")
    input("按回车键退出...")


if __name__ == "__main__":
    main()
