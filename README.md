# RAG系统 - 基于Word2Vec、Faiss和Qwen的检索增强生成系统

这是一个完整的RAG（Retrieval-Augmented Generation）系统，集成了Word2Vec嵌入模型、Faiss向量数据库和Qwen大语言模型。

## 系统架构
RAG系统
├── Word2Vec嵌入模型 (word2vec_model.py)
│   ├── 文本预处理
│   ├── 模型训练
│   └── 向量生成
├── Faiss向量数据库 (vector_database.py)
│   ├── 向量索引
│   ├── 相似度搜索
│   └── 文档存储
├── Qwen大语言模型 (qwen_llm.py)
│   ├── 文本生成
│   ├── 问答系统
│   └── 文档重排序
└── RAG主系统 (rag_system.py)
    ├── 知识库构建
    ├── 查询处理
    └── 答案生成

## 快速安装

### 自动安装（推荐）

#### Windows用户
```batch
# 克隆或下载项目后，运行自动安装脚本
install.bat
```

#### Linux/Mac用户
```bash
# 给脚本添加执行权限并运行
chmod +x install.sh
./install.sh
```

### 手动安装

#### 1. 环境要求
- Python 3.8 或更高版本
- pip 包管理器
- 至少 4GB 可用内存
- （可选）NVIDIA GPU 用于加速

#### 2. 创建虚拟环境
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### 3. 安装依赖
```bash
# 升级pip
python -m pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

#### 4. 运行安装脚本
```bash
# 创建目录结构和配置文件
python install_setup.py
```

#### 5. 环境检查
```bash
# 验证安装是否成功
python check_environment.py
```

## 环境配置

### 1. API密钥配置

编辑 `.env` 文件，设置您的Qwen API密钥：

```bash
# 编辑 .env 文件
DASHSCOPE_API_KEY=your_qwen_api_key_here
```

或者设置环境变量：

```bash
# Windows
set DASHSCOPE_API_KEY=your_api_key_here

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

### 2. 获取Qwen API密钥

1. 访问 [阿里云DashScope控制台](https://dashscope.console.aliyun.com/)
2. 注册/登录账号
3. 创建API密钥
4. 将密钥复制到配置文件中

### 3. 数据准备

将您的英文文本文件放入 `data/raw_texts/` 目录：

## 功能特性

### 1. Word2Vec嵌入模型
- **强大的英文文本处理**: 专门针对英文文档优化
- **自定义预处理**: 停用词过滤、词干提取、标准化
- **灵活的模型配置**: 可调整向量维度、窗口大小、训练轮数等
- **句子级嵌入**: 通过词向量平均生成句子嵌入
- **相似词查找**: 支持语义相似词检索

### 2. Faiss向量数据库
- **多种索引类型**: 支持Flat、IVF、HNSW等索引
- **高效检索**: 毫秒级向量相似度搜索
- **可扩展性**: 支持大规模向量数据
- **持久化存储**: 支持模型和数据的保存/加载
- **元数据管理**: 完整的文档元信息存储

### 3. Qwen大语言模型
- **多模型支持**: qwen-turbo、qwen-plus、qwen-max
- **RAG优化**: 专门的RAG提示模板
- **文档重排序**: 基于LLM的相关性评估
- **多种任务**: 问答、摘要、关键词提取
- **可配置参数**: 温度、top-p、最大长度等

### 4. 完整RAG流程
- **智能文档分块**: 可配置的分块大小和重叠
- **端到端处理**: 从文档加载到答案生成
- **质量控制**: 相似度阈值、置信度评估
- **增量更新**: 支持动态添加新文档
- **系统监控**: 详细的统计信息和日志

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境配置

设置阿里云DashScope API密钥：

```bash
# Windows
set DASHSCOPE_API_KEY=your_api_key_here

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

## 快速开始

### 1. 运行演示

```python
python demo.py
```

### 2. 基本使用

```python
from rag_system import RAGSystem

# 初始化RAG系统
rag = RAGSystem(
    vector_size=300,
    index_type="flat",
    qwen_model="qwen-turbo"
)

# 构建知识库
file_paths = ["doc1.txt", "doc2.txt", "doc3.txt"]
rag.build_knowledge_base(file_paths, save_path="my_rag")

# 查询
result = rag.query("What is machine learning?")
print("答案:", result['answer'])
print("置信度:", result['confidence'])

# 保存系统
rag.save_system("my_rag_system")
```

### 3. 高级使用

```python
# 自定义配置
rag = RAGSystem(
    vector_size=512,           # 更大的向量维度
    index_type="hnsw",         # 更快的索引类型
    qwen_model="qwen-max"      # 更强的模型
)

# 高级查询
result = rag.query(
    question="Explain deep learning",
    top_k=10,                  # 检索更多文档
    similarity_threshold=0.5,  # 更高的相似度阈值
    use_rerank=True           # 启用LLM重排序
)

# 添加新文档
rag.add_document(
    content="New document content...",
    metadata={"source": "manual_input", "topic": "AI"}
)
```

## 文件说明

- `word2vec_model.py`: Word2Vec嵌入模型实现
- `vector_database.py`: Faiss向量数据库封装
- `qwen_llm.py`: Qwen大语言模型接口
- `rag_system.py`: RAG系统主类
- `demo.py`: 完整的演示程序
- `requirements.txt`: 项目依赖

## 配置参数

### Word2Vec参数
- `vector_size`: 词向量维度 (默认: 300)
- `window`: 上下文窗口大小 (默认: 5)
- `min_count`: 最小词频 (默认: 2)
- `epochs`: 训练轮数 (默认: 100)

### Faiss参数
- `index_type`: 索引类型 ("flat", "ivf", "hnsw")
- `dimension`: 向量维度

### 文档处理参数
- `chunk_size`: 文档分块大小 (默认: 500词)
- `chunk_overlap`: 分块重叠大小 (默认: 50词)

### 查询参数
- `top_k`: 检索文档数量 (默认: 5)
- `similarity_threshold`: 相似度阈值 (默认: 0.3)
- `use_rerank`: 是否使用LLM重排序 (默认: True)

## 性能优化建议

### 1. 向量维度选择
- 小数据集: 100-200维
- 中等数据集: 300-512维
- 大数据集: 512-1024维

### 2. 索引类型选择
- 小数据集(<10K): "flat"
- 中等数据集(10K-1M): "ivf"
- 大数据集(>1M): "hnsw"

### 3. 文档分块策略
- 技术文档: 300-500词/块
- 长篇文章: 500-800词/块
- 对话数据: 100-300词/块

## 常见问题

### Q: 如何处理中文文档？
A: 当前系统专门针对英文优化。处理中文需要修改预处理逻辑，使用jieba分词等。

### Q: 如何提高检索准确性？
A: 
1. 增加训练数据量
2. 调整Word2Vec参数
3. 启用LLM重排序
4. 优化文档分块策略

### Q: 系统支持多大规模的数据？
A: 
- Word2Vec: 支持GB级文本训练
- Faiss: 支持百万到十亿级向量
- 整体性能取决于硬件配置

### Q: 如何减少内存使用？
A: 
1. 使用较小的向量维度
2. 选择内存友好的索引类型
3. 分批处理大文档

## 扩展功能

系统支持以下扩展：

1. **多模态支持**: 添加图像、音频处理
2. **实时更新**: 支持文档的实时添加/删除
3. **分布式部署**: 支持多机部署
4. **API服务**: 提供REST API接口
5. **Web界面**: 添加用户友好的Web界面

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！