#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统安装设置脚本
用于下载必要的模型和数据，创建目录结构
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import ssl
from pathlib import Path

def print_step(step_name):
    """打印安装步骤"""
    print(f"\n{'='*50}")
    print(f"正在执行: {step_name}")
    print(f"{'='*50}")

def create_directories():
    """创建必要的目录结构"""
    print_step("创建目录结构")
    
    directories = [
        "data",
        "data/raw_texts",
        "data/processed",
        "models",
        "models/word2vec",
        "models/faiss_index",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {directory}")

def download_nltk_data():
    """下载NLTK数据"""
    print_step("下载NLTK数据")
    
    try:
        import nltk
        
        # 设置NLTK数据路径
        nltk_data_dir = Path("data/nltk_data")
        nltk_data_dir.mkdir(exist_ok=True)
        nltk.data.path.append(str(nltk_data_dir))
        
        # 下载必要的NLTK数据
        datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for dataset in datasets:
            try:
                print(f"下载 {dataset}...")
                nltk.download(dataset, download_dir=str(nltk_data_dir))
                print(f"✓ {dataset} 下载完成")
            except Exception as e:
                print(f"⚠ {dataset} 下载失败: {e}")
                
    except ImportError:
        print("⚠ NLTK未安装，跳过NLTK数据下载")

def create_sample_data():
    """创建示例数据文件"""
    print_step("创建示例数据")
    
    sample_texts = [
        """
        Artificial Intelligence (AI) is intelligence demonstrated by machines, 
        in contrast to the natural intelligence displayed by humans and animals. 
        Leading AI textbooks define the field as the study of "intelligent agents": 
        any device that perceives its environment and takes actions that maximize 
        its chance of successfully achieving its goals.
        """,
        """
        Machine Learning is a method of data analysis that automates analytical 
        model building. It is a branch of artificial intelligence based on the 
        idea that systems can learn from data, identify patterns and make 
        decisions with minimal human intervention.
        """,
        """
        Natural Language Processing (NLP) is a subfield of linguistics, computer 
        science, and artificial intelligence concerned with the interactions 
        between computers and human language, in particular how to program 
        computers to process and analyze large amounts of natural language data.
        """,
        """
        Deep Learning is part of a broader family of machine learning methods 
        based on artificial neural networks with representation learning. 
        Learning can be supervised, semi-supervised or unsupervised.
        """,
        """
        Computer Vision is an interdisciplinary scientific field that deals 
        with how computers can gain high-level understanding from digital 
        images or videos. From the perspective of engineering, it seeks to 
        understand and automate tasks that the human visual system can do.
        """
    ]
    
    # 创建示例文本文件
    for i, text in enumerate(sample_texts, 1):
        file_path = Path(f"data/raw_texts/sample_{i}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text.strip())
        print(f"✓ 创建示例文件: {file_path}")

def create_config_file():
    """创建配置文件"""
    print_step("创建配置文件")
    
    config_content = """# RAG系统配置文件
# Word2Vec模型配置
WORD2VEC_VECTOR_SIZE = 300
WORD2VEC_WINDOW = 5
WORD2VEC_MIN_COUNT = 1
WORD2VEC_WORKERS = 4
WORD2VEC_SG = 1  # 1 for skip-gram, 0 for CBOW

# Faiss索引配置
FAISS_INDEX_TYPE = "IndexFlatIP"  # 内积索引
FAISS_NLIST = 100  # 用于IVF索引的聚类数量

# Qwen模型配置
QWEN_MODEL_NAME = "qwen-turbo"
QWEN_API_KEY = "your_api_key_here"  # 请替换为您的API密钥

# 文本处理配置
MAX_CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 100

# 检索配置
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.7

# 日志配置
LOG_LEVEL = "INFO"
LOG_FILE = "logs/rag_system.log"
"""
    
    with open("config.py", 'w', encoding='utf-8') as f:
        f.write(config_content)
    print("✓ 创建配置文件: config.py")

def create_env_file():
    """创建环境变量文件"""
    print_step("创建环境变量文件")
    
    env_content = """# RAG系统环境变量
# 请根据实际情况修改以下配置

# Qwen API配置
DASHSCOPE_API_KEY=your_qwen_api_key_here

# 模型路径配置
WORD2VEC_MODEL_PATH=models/word2vec/word2vec_model.bin
FAISS_INDEX_PATH=models/faiss_index/document_index.faiss
FAISS_METADATA_PATH=models/faiss_index/metadata.json

# 数据路径配置
RAW_TEXT_DIR=data/raw_texts
PROCESSED_DATA_DIR=data/processed
LOG_DIR=logs

# 系统配置
PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0
"""
    
    with open(".env", 'w', encoding='utf-8') as f:
        f.write(env_content)
    print("✓ 创建环境变量文件: .env")

def create_startup_script():
    """创建启动脚本"""
    print_step("创建启动脚本")
    
    # Windows启动脚本
    startup_bat = """@echo off
chcp 65001 >nul
echo 启动RAG系统...

if not exist "venv\\Scripts\\activate.bat" (
    echo 错误: 虚拟环境不存在，请先运行 install.bat
    pause
    exit /b 1
)

call venv\\Scripts\\activate.bat
python demo.py
pause
"""
    
    with open("start.bat", 'w', encoding='utf-8') as f:
        f.write(startup_bat)
    print("✓ 创建Windows启动脚本: start.bat")
    
    # Linux/Mac启动脚本
    startup_sh = """#!/bin/bash
echo "启动RAG系统..."

if [ ! -f "venv/bin/activate" ]; then
    echo "错误: 虚拟环境不存在，请先运行安装脚本"
    exit 1
fi

source venv/bin/activate
python demo.py
"""
    
    with open("start.sh", 'w', encoding='utf-8') as f:
        f.write(startup_sh)
    
    # 给shell脚本添加执行权限
    try:
        os.chmod("start.sh", 0o755)
        print("✓ 创建Linux/Mac启动脚本: start.sh")
    except:
        print("⚠ 无法设置start.sh执行权限（Windows系统正常）")

def check_dependencies():
    """检查依赖是否正确安装"""
    print_step("检查依赖安装")
    
    required_packages = [
        'gensim', 'faiss', 'numpy', 'sklearn', 
        'nltk', 'transformers', 'torch', 'dashscope'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'faiss':
                import faiss
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ 以下包未正确安装: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ 所有依赖包已正确安装")
        return True

def main():
    """主安装函数"""
    print("RAG系统安装设置脚本")
    print("=" * 50)
    
    try:
        # 创建目录结构
        create_directories()
        
        # 下载NLTK数据
        download_nltk_data()
        
        # 创建示例数据
        create_sample_data()
        
        # 创建配置文件
        create_config_file()
        
        # 创建环境变量文件
        create_env_file()
        
        # 创建启动脚本
        create_startup_script()
        
        # 检查依赖
        deps_ok = check_dependencies()
        
        print_step("安装完成")
        print("✓ RAG系统环境设置完成！")
        print("\n下一步操作:")
        print("1. 编辑 .env 文件，设置您的Qwen API密钥")
        print("2. 将您的英文文本文件放入 data/raw_texts/ 目录")
        print("3. 运行 python demo.py 开始使用系统")
        print("4. 或者运行 start.bat (Windows) / start.sh (Linux/Mac)")
        
        if not deps_ok:
            print("\n⚠ 注意: 部分依赖包未正确安装，请检查并重新安装")
            
    except Exception as e:
        print(f"\n❌ 安装过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()