#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检查脚本
用于验证RAG系统的环境是否正确配置
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print("需要Python 3.8或更高版本")
        return False

def check_packages():
    """检查必要的包"""
    print("\n检查Python包...")
    
    packages = {
        'gensim': 'Word2Vec模型训练',
        'faiss': '向量数据库',
        'numpy': '数值计算',
        'sklearn': '机器学习工具',
        'nltk': '自然语言处理',
        'transformers': 'Transformer模型',
        'torch': 'PyTorch深度学习框架',
        'dashscope': 'Qwen模型API',
        'tqdm': '进度条显示',
        'jieba': '中文分词'
    }
    
    missing_packages = []
    
    for package, description in packages.items():
        try:
            if package == 'faiss':
                import faiss
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package:<15} - {description}")
        except ImportError:
            print(f"✗ {package:<15} - {description} (未安装)")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_directories():
    """检查目录结构"""
    print("\n检查目录结构...")
    
    required_dirs = [
        "data",
        "data/raw_texts",
        "data/processed",
        "models",
        "models/word2vec",
        "models/faiss_index",
        "logs"
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✓ {directory}")
        else:
            print(f"✗ {directory} (不存在)")
            missing_dirs.append(directory)
    
    return len(missing_dirs) == 0, missing_dirs

def check_config_files():
    """检查配置文件"""
    print("\n检查配置文件...")
    
    config_files = {
        "config.py": "系统配置文件",
        ".env": "环境变量文件",
        "requirements.txt": "依赖包列表"
    }
    
    missing_files = []
    
    for file_path, description in config_files.items():
        if Path(file_path).exists():
            print(f"✓ {file_path:<20} - {description}")
        else:
            print(f"✗ {file_path:<20} - {description} (不存在)")
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def check_sample_data():
    """检查示例数据"""
    print("\n检查示例数据...")
    
    data_dir = Path("data/raw_texts")
    if not data_dir.exists():
        print("✗ 数据目录不存在")
        return False
    
    txt_files = list(data_dir.glob("*.txt"))
    if txt_files:
        print(f"✓ 找到 {len(txt_files)} 个文本文件:")
        for file_path in txt_files[:5]:  # 只显示前5个
            print(f"  - {file_path.name}")
        if len(txt_files) > 5:
            print(f"  ... 还有 {len(txt_files) - 5} 个文件")
        return True
    else:
        print("⚠ 数据目录为空，请添加英文文本文件")
        return False

def check_gpu_support():
    """检查GPU支持"""
    print("\n检查GPU支持...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ 检测到 {gpu_count} 个GPU: {gpu_name}")
            print(f"✓ CUDA版本: {torch.version.cuda}")
            return True
        else:
            print("⚠ 未检测到GPU，将使用CPU模式")
            return False
    except ImportError:
        print("⚠ PyTorch未安装，无法检查GPU支持")
        return False

def main():
    """主检查函数"""
    print("RAG系统环境检查")
    print("=" * 50)
    
    checks = []
    
    # 检查Python版本
    checks.append(check_python_version())
    
    # 检查包安装
    packages_ok, missing_packages = check_packages()
    checks.append(packages_ok)
    
    # 检查目录结构
    dirs_ok, missing_dirs = check_directories()
    checks.append(dirs_ok)
    
    # 检查配置文件
    config_ok, missing_files = check_config_files()
    checks.append(config_ok)
    
    # 检查示例数据
    data_ok = check_sample_data()
    
    # 检查GPU支持
    gpu_ok = check_gpu_support()
    
    # 总结
    print("\n" + "=" * 50)
    print("环境检查总结")
    print("=" * 50)
    
    if all(checks):
        print("✓ 环境检查通过！系统已准备就绪。")
        
        if not data_ok:
            print("\n建议:")
            print("- 将您的英文文本文件放入 data/raw_texts/ 目录")
        
        if not gpu_ok:
            print("- 考虑安装GPU版本的PyTorch以获得更好性能")
            
        print("\n可以运行以下命令开始使用:")
        print("python demo.py")
        
    else:
        print("✗ 环境检查失败，需要解决以下问题:")
        
        if missing_packages:
            print(f"\n缺失的包: {', '.join(missing_packages)}")
            print("运行: pip install -r requirements.txt")
        
        if missing_dirs:
            print(f"\n缺失的目录: {', '.join(missing_dirs)}")
            print("运行: python install_setup.py")
        
        if missing_files:
            print(f"\n缺失的文件: {', '.join(missing_files)}")
            print("运行: python install_setup.py")

if __name__ == "__main__":
    main()