@echo off
chcp 65001 >nul
echo ========================================
echo RAG系统环境安装脚本
echo ========================================
echo.

echo 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python环境，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python环境检查通过！
echo.

echo 检查pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到pip，请检查Python安装
    pause
    exit /b 1
)

echo pip检查通过！
echo.

echo 创建虚拟环境...
if not exist "venv" (
    python -m venv venv
    echo 虚拟环境创建成功！
) else (
    echo 虚拟环境已存在，跳过创建
)

echo.
echo 激活虚拟环境...
call venv\Scripts\activate.bat

echo.
echo 升级pip...
python -m pip install --upgrade pip

echo.
echo 安装项目依赖...
pip install -r requirements.txt

echo.
echo 运行Python安装脚本...
python install_setup.py

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 使用方法:
echo 1. 激活虚拟环境: venv\Scripts\activate.bat
echo 2. 运行演示: python demo.py
echo 3. 退出虚拟环境: deactivate
echo.
pause