#!/bin/bash

echo "========================================"
echo "RAG系统环境安装脚本 (Linux/Mac)"
echo "========================================"
echo

# 检查Python环境
echo "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3环境，请先安装Python 3.8+"
    echo "Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
    echo "CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "macOS: brew install python3"
    exit 1
fi

echo "Python环境检查通过！"
echo

# 检查pip
echo "检查pip..."
if ! command -v pip3 &> /dev/null; then
    echo "错误: 未找到pip3，请检查Python安装"
    exit 1
fi

echo "pip检查通过！"
echo

# 创建虚拟环境
echo "创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "虚拟环境创建成功！"
else
    echo "虚拟环境已存在，跳过创建"
fi

echo
echo "激活虚拟环境..."
source venv/bin/activate

echo
echo "升级pip..."
python -m pip install --upgrade pip

echo
echo "安装项目依赖..."
pip install -r requirements.txt

echo
echo "运行Python安装脚本..."
python install_setup.py

echo
echo "========================================"
echo "安装完成！"
echo "========================================"
echo
echo "使用方法:"
echo "1. 激活虚拟环境: source venv/bin/activate"
echo "2. 运行演示: python demo.py"
echo "3. 退出虚拟环境: deactivate"
echo

# 设置执行权限
chmod +x start.sh

echo "提示: 可以运行 ./start.sh 快速启动系统"