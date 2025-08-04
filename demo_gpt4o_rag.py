import os
import logging
import re
from rag_system_openai import RAGSystem
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# 设置日志级别
logging.basicConfig(level=logging.INFO)

def create_enhanced_sample_documents():
    """创建增强的示例文档，针对.txt文件优化"""
    
    # 示例文档1：人工智能技术指南
    doc1_content = """
# 人工智能技术指南

## 1. 机器学习基础

### 1.1 什么是机器学习？
机器学习是人工智能的一个重要分支，它使计算机系统能够从数据中自动学习和改进，而无需明确编程。机器学习的核心思想是通过算法分析大量数据，识别模式，并基于这些模式做出预测或决策。

### 1.2 机器学习的类型

#### 监督学习
监督学习使用标记的训练数据来学习输入和输出之间的映射关系。主要包括：
- 分类问题：预测离散的类别标签
- 回归问题：预测连续的数值

常用算法：
- 线性回归和逻辑回归
- 决策树和随机森林
- 支持向量机（SVM）
- 神经网络

#### 无监督学习
无监督学习从未标记的数据中发现隐藏的模式和结构。主要包括：
- 聚类：将相似的数据点分组
- 降维：减少数据的特征数量
- 关联规则挖掘：发现数据项之间的关系

常用算法：
- K-means聚类
- 层次聚类
- 主成分分析（PCA）
- 独立成分分析（ICA）

#### 强化学习
强化学习通过与环境交互来学习最优策略。智能体通过执行动作获得奖励或惩罚，目标是最大化累积奖励。

关键概念：
- 智能体（Agent）：学习和决策的实体
- 环境（Environment）：智能体操作的世界
- 状态（State）：环境的当前情况
- 动作（Action）：智能体可以执行的操作
- 奖励（Reward）：环境给予的反馈

### 1.3 机器学习的应用领域

#### 医疗健康
- 医学影像诊断：使用深度学习分析X光、CT、MRI图像
- 药物发现：预测分子特性，加速新药开发
- 个性化治疗：基于患者数据制定治疗方案
- 疾病预测：分析电子健康记录预测疾病风险

#### 金融服务
- 风险评估：评估贷款和投资风险
- 欺诈检测：识别可疑交易和行为
- 算法交易：自动化投资决策
- 信用评分：评估客户信用状况

#### 科技互联网
- 搜索引擎：改进搜索结果的相关性
- 推荐系统：个性化内容和产品推荐
- 计算机视觉：图像识别和处理
- 自然语言处理：机器翻译、聊天机器人

#### 交通运输
- 自动驾驶：感知、决策和控制
- 路线优化：智能导航和交通管理
- 预测性维护：预测车辆和基础设施故障
- 需求预测：优化公共交通调度

## 2. 深度学习进阶

### 2.1 神经网络基础
神经网络是受生物神经系统启发的计算模型，由相互连接的节点（神经元）组成。

#### 基本组件
- 神经元：接收输入，应用权重和偏置，通过激活函数产生输出
- 层：神经元的集合，包括输入层、隐藏层和输出层
- 权重：连接强度的参数
- 偏置：帮助模型更好拟合数据的参数
- 激活函数：引入非线性的数学函数

#### 常用激活函数
1. Sigmoid：将输入映射到0-1之间
   - 优点：输出范围有界，平滑可导
   - 缺点：梯度消失问题，计算复杂

2. ReLU（修正线性单元）：负值输出0，正值保持不变
   - 优点：计算简单，缓解梯度消失
   - 缺点：可能导致神经元死亡

3. Tanh：将输入映射到-1到1之间
   - 优点：零中心化，收敛更快
   - 缺点：仍存在梯度消失问题

4. Leaky ReLU：负值乘以小的正数
   - 优点：避免神经元死亡
   - 缺点：需要调整参数

### 2.2 卷积神经网络（CNN）
CNN专门用于处理具有网格结构的数据，如图像。

#### 核心组件
- 卷积层：使用滤波器检测局部特征
- 池化层：降低空间维度，保留重要信息
- 全连接层：传统神经网络层，用于最终分类

#### 关键概念
- 滤波器/卷积核：检测特定特征的权重矩阵
- 步长（Stride）：滤波器移动的距离
- 填充（Padding）：在输入边缘添加值
- 特征图：卷积操作的输出

#### CNN应用
- 图像分类：识别图像中的对象类别
- 目标检测：定位和识别图像中的多个对象
- 语义分割：为图像中每个像素分配类别标签
- 人脸识别：识别和验证人脸身份

### 2.3 循环神经网络（RNN）
RNN专门处理序列数据，具有记忆能力。

#### RNN变体
1. 标准RNN：基本的循环架构
   - 优点：能处理变长序列
   - 缺点：梯度消失问题

2. LSTM（长短期记忆网络）：解决长期依赖问题
   - 遗忘门：决定丢弃哪些信息
   - 输入门：决定存储哪些新信息
   - 输出门：决定输出哪些信息

3. GRU（门控循环单元）：LSTM的简化版本
   - 更新门：控制信息更新
   - 重置门：控制历史信息的使用

#### RNN应用
- 自然语言处理：文本生成、机器翻译
- 语音识别：将音频转换为文本
- 时间序列预测：股价、天气预测
- 情感分析：分析文本情感倾向

### 2.4 Transformer架构
Transformer彻底改变了自然语言处理领域。

#### 核心机制
- 自注意力机制：计算序列中每个位置与其他位置的关系
- 多头注意力：并行计算多个注意力表示
- 位置编码：为序列添加位置信息
- 前馈网络：非线性变换

#### 优势
- 并行计算：比RNN更高效
- 长距离依赖：更好地处理长序列
- 可解释性：注意力权重提供解释
- 迁移学习：预训练模型可以微调

#### 著名模型
- BERT：双向编码器表示
- GPT：生成式预训练Transformer
- T5：文本到文本转换Transformer
- BART：去噪自编码器

## 3. 模型训练和优化

### 3.1 训练过程
1. 前向传播：数据通过网络产生预测
2. 损失计算：衡量预测与真实值的差异
3. 反向传播：计算梯度
4. 参数更新：使用优化算法更新权重

### 3.2 优化算法
- SGD（随机梯度下降）：基础优化算法
- Adam：自适应学习率优化器
- RMSprop：适合处理非平稳目标
- AdaGrad：适合稀疏数据

### 3.3 正则化技术
- Dropout：随机关闭神经元防止过拟合
- 批量归一化：标准化层输入
- 权重衰减：L1/L2正则化
- 早停：监控验证集性能

### 3.4 评估指标
- 分类：准确率、精确率、召回率、F1分数
- 回归：均方误差、平均绝对误差、R²
- 排序：NDCG、MAP、MRR

这些技术和概念构成了现代人工智能的基础，理解它们对于开发有效的AI系统至关重要。
"""

    # 示例文档2：自然语言处理实践指南
    doc2_content = """
# 自然语言处理实践指南

## 1. NLP基础概念

### 1.1 什么是自然语言处理？
自然语言处理（Natural Language Processing, NLP）是计算机科学、人工智能和语言学的交叉领域，专注于让计算机理解、解释和生成人类语言。NLP的目标是弥合人类语言和计算机理解之间的差距。

### 1.2 NLP的挑战
人类语言具有以下特点，使得NLP变得复杂：

#### 歧义性
- 词汇歧义：一个词有多个含义
- 句法歧义：句子结构可以有多种解释
- 语义歧义：句子含义不明确
- 语用歧义：依赖上下文的含义

#### 复杂性
- 语法规则复杂且有例外
- 习语和俚语的使用
- 文化和地域差异
- 语言的动态演变

#### 上下文依赖
- 代词指代
- 省略和暗示
- 讽刺和幽默
- 情感和语调

## 2. 文本预处理

### 2.1 基础预处理步骤

#### 文本清理
```python
import re
import string

def clean_text(text):
    # 转换为小写
    text = text.lower()
    
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 移除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 移除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

#### 分词（Tokenization）
分词是将文本分解为更小单位的过程：

1. 词级分词：将文本分解为单词
2. 句子分词：将文本分解为句子
3. 子词分词：将单词分解为更小的单位

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# 下载必要的数据
nltk.download('punkt')

text = "Hello world. This is a sample text."

# 词分词
words = word_tokenize(text)
print("Words:", words)

# 句子分词
sentences = sent_tokenize(text)
print("Sentences:", sentences)
```

#### 停用词移除
停用词是在文本中频繁出现但通常不携带重要语义信息的词：

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)
```

#### 词干提取和词形还原
- 词干提取（Stemming）：将词汇还原到词根形式
- 词形还原（Lemmatization）：将词汇还原到词典形式

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('wordnet')

# 词干提取
stemmer = PorterStemmer()
print(stemmer.stem("running"))  # run

# 词形还原
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos='v'))  # run
```

### 2.2 高级预处理技术

#### 拼写纠错
```python
from textblob import TextBlob

def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())

text = "I havv goood speling"
corrected = correct_spelling(text)
print(corrected)  # I have good spelling
```

#### 文本标准化
- 数字标准化：将数字转换为统一格式
- 缩写展开：将缩写形式展开为完整形式
- 大小写标准化：统一大小写格式

## 3. 特征提取

### 3.1 传统方法

#### 词袋模型（Bag of Words）
```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love machine learning",
    "Machine learning is awesome",
    "I love programming"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
print("Feature names:", vectorizer.get_feature_names_out())
print("BOW matrix:\n", X.toarray())
```

#### TF-IDF（词频-逆文档频率）
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
print("TF-IDF matrix:\n", X.toarray())
```

#### N-gram模型
```python
# 二元语法（Bigram）
vectorizer = CountVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(documents)
print("Bigram features:", vectorizer.get_feature_names_out())
```

### 3.2 现代方法

#### Word2Vec
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# 准备数据
sentences = [word_tokenize(doc.lower()) for doc in documents]

# 训练Word2Vec模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 获取词向量
vector = model.wv['machine']
print("Vector for 'machine':", vector[:10])  # 显示前10个维度

# 找相似词
similar_words = model.wv.most_similar('machine', topn=3)
print("Similar words:", similar_words)
```

#### BERT嵌入
```python
from transformers import BertTokenizer, BertModel
import torch

# 加载预训练模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    # 编码文本
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # 获取嵌入
    with torch.no_grad():
        outputs = model(**encoded)
        embeddings = outputs.last_hidden_state
    
    # 返回[CLS]标记的嵌入
    return embeddings[0][0].numpy()

text = "This is a sample sentence."
embedding = get_bert_embedding(text)
print("BERT embedding shape:", embedding.shape)
```

## 4. 核心NLP任务

### 4.1 文本分类

#### 情感分析
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 示例数据
texts = [
    "I love this movie!",
    "This movie is terrible.",
    "Great acting and plot.",
    "Boring and predictable."
]
labels = [1, 0, 1, 0]  # 1: positive, 0: negative

# 特征提取
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# 训练模型
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 预测
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
```

#### 主题分类
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB

# 加载数据
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

# 特征提取
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = vectorizer.fit_transform(newsgroups_train.data)

# 训练分类器
classifier = MultinomialNB()
classifier.fit(X_train, newsgroups_train.target)

# 测试
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
X_test = vectorizer.transform(newsgroups_test.data)
predictions = classifier.predict(X_test)

accuracy = (predictions == newsgroups_test.target).mean()
print(f"Accuracy: {accuracy:.3f}")
```

### 4.2 命名实体识别（NER）

#### 使用spaCy进行NER
```python
import spacy

# 加载模型
nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
doc = nlp(text)

# 提取实体
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}, Description: {spacy.explain(ent.label_)}")
```

#### 自定义NER模型
```python
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

def word_features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    
    return features

def sent_features(sent):
    return [word_features(sent, i) for i in range(len(sent))]

def sent_labels(sent):
    return [label for token, label in sent]

# 训练CRF模型
crf = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

# 假设有训练数据 train_sents
# X_train = [sent_features(s) for s in train_sents]
# y_train = [sent_labels(s) for s in train_sents]
# crf.fit(X_train, y_train)
```

### 4.3 机器翻译

#### 使用Transformer进行翻译
```python
from transformers import MarianMTModel, MarianTokenizer

# 加载预训练的翻译模型（英语到德语）
model_name = 'Helsinki-NLP/opus-mt-en-de'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text, model, tokenizer):
    # 编码输入文本
    inputs = tokenizer.encode(text, return_tensors="pt", padding=True)
    
    # 生成翻译
    translated = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    
    # 解码输出
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# 翻译示例
english_text = "Hello, how are you today?"
german_translation = translate_text(english_text, model, tokenizer)
print(f"English: {english_text}")
print(f"German: {german_translation}")
```

### 4.4 问答系统

#### 基于检索的问答
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleQASystem:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = self.vectorizer.fit_transform(documents)
    
    def answer_question(self, question, top_k=3):
        # 向量化问题
        question_vector = self.vectorizer.transform([question])
        
        # 计算相似度
        similarities = cosine_similarity(question_vector, self.doc_vectors).flatten()
        
        # 获取最相似的文档
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'similarity': similarities[idx]
            })
        
        return results

# 示例使用
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing helps computers understand human language.",
    "Computer vision enables machines to interpret visual information."
]

qa_system = SimpleQASystem(documents)
question = "What is machine learning?"
answers = qa_system.answer_question(question)

for i, answer in enumerate(answers):
    print(f"Answer {i+1}: {answer['document']} (Similarity: {answer['similarity']:.3f})")
```

## 5. 评估指标

### 5.1 分类任务指标
- **准确率（Accuracy）**：正确预测的比例
- **精确率（Precision）**：预测为正类中实际为正类的比例
- **召回率（Recall）**：实际正类中被正确预测的比例
- **F1分数**：精确率和召回率的调和平均数

### 5.2 序列标注指标
- **实体级F1**：完全匹配实体的F1分数
- **标记级F1**：单个标记的F1分数

### 5.3 生成任务指标
- **BLEU**：机器翻译质量评估
- **ROUGE**：文本摘要质量评估
- **METEOR**：考虑同义词的评估指标

### 5.4 语言模型指标
- **困惑度（Perplexity）**：模型对测试数据的不确定性
- **交叉熵损失**：预测分布与真实分布的差异

## 6. 实际应用案例

### 6.1 聊天机器人开发
```python
import random

class SimpleChatbot:
    def __init__(self):
        self.responses = {
            'greeting': ['Hello!', 'Hi there!', 'Hey!'],
            'goodbye': ['Goodbye!', 'See you later!', 'Bye!'],
            'default': ['I don\'t understand.', 'Can you rephrase?', 'Tell me more.']
        }
        
        self.patterns = {
            'greeting': ['hello', 'hi', 'hey'],
            'goodbye': ['bye', 'goodbye', 'see you']
        }
    
    def classify_intent(self, message):
        message = message.lower()
        for intent, patterns in self.patterns.items():
            if any(pattern in message for pattern in patterns):
                return intent
        return 'default'
    
    def generate_response(self, message):
        intent = self.classify_intent(message)
        return random.choice(self.responses[intent])

# 使用示例
chatbot = SimpleChatbot()
print(chatbot.generate_response("Hello there!"))
print(chatbot.generate_response("Goodbye!"))
```

### 6.2 文档摘要系统
```python
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def summarize_text(text, sentences_count=3, language="english"):
    # 解析文本
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    
    # 创建摘要器
    stemmer = Stemmer(language)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    
    # 生成摘要
    summary = summarizer(parser.document, sentences_count)
    
    return [str(sentence) for sentence in summary]

# 示例使用
long_text = "
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language. The goal of NLP is to enable computers to understand, interpret 
and manipulate human language. This involves several challenges including speech 
recognition, natural language understanding, and natural language generation.
"

summary = summarize_text(long_text, 2)
print("Summary:")
for sentence in summary:
    print(f"- {sentence}")

# 示例使用
long_text = "
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language. The goal of NLP is to enable computers to understand, interpret 
and manipulate human language. This involves several challenges including speech 
recognition, natural language understanding, and natural language generation.


summary = summarize_text(long_text, 2)
print("Summary:")
for sentence in summary:
    print(f"- {sentence}")
```

这个指南涵盖了NLP的核心概念、技术和实际应用，为开发高质量的NLP系统提供了全面的基础。
"""

    # 示例文档3：RAG系统设计与实现
    doc3_content = """
# RAG系统设计与实现指南

## 1. RAG系统概述

### 1.1 什么是RAG？
检索增强生成（Retrieval-Augmented Generation, RAG）是一种结合了信息检索和文本生成的AI架构。RAG系统通过检索相关文档来增强大语言模型的生成能力，使其能够基于外部知识库回答问题。

### 1.2 RAG系统的优势
- **知识更新**：无需重新训练模型即可更新知识
- **可解释性**：可以追溯答案来源
- **准确性**：基于事实文档生成答案
- **成本效益**：比训练大模型更经济
- **领域适应**：容易适应特定领域

### 1.3 RAG vs 传统方法

#### 传统问答系统
- 基于规则的系统：依赖手工制定的规则
- 基于模板的系统：使用预定义模板
- 基于检索的系统：直接返回相关文档

#### 端到端神经网络
- 需要大量训练数据
- 难以更新知识
- 可能产生幻觉
- 缺乏可解释性

#### RAG系统
- 结合检索和生成的优势
- 动态知识更新
- 更好的事实准确性
- 提供答案来源

## 2. RAG系统架构

### 2.1 核心组件

#### 文档处理器
负责处理和预处理输入文档：
```python
class DocumentProcessor:
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process_document(self, text):
        # 清理文本
        cleaned_text = self.clean_text(text)
        
        # 分块
        chunks = self.chunk_text(cleaned_text)
        
        # 提取元数据
        metadata = self.extract_metadata(text)
        
        return chunks, metadata
    
    def clean_text(self, text):
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        return text.strip()
    
    def chunk_text(self, text):
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        
        return chunks
```

#### 嵌入模型
将文本转换为向量表示：
```python
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts):
        return self.model.encode(texts)
    
    def encode_query(self, query):
        return self.model.encode([query])[0]
```

#### 向量数据库
存储和检索文档向量：
```python
import faiss
import numpy as np

class VectorDatabase:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # 内积索引
        self.documents = []
        self.metadata = []
    
    def add_documents(self, embeddings, documents, metadata):
        # 标准化向量
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding, k=5):
        # 标准化查询向量
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score)
                })
        
        return results
```

#### 生成模型
基于检索到的文档生成答案：
```python
from transformers import pipeline

class GenerationModel:
    def __init__(self, model_name='gpt-3.5-turbo'):
        self.model_name = model_name
        if 'gpt' in model_name:
            import openai
            self.client = openai.OpenAI()
        else:
            self.generator = pipeline('text-generation', model=model_name)
    
    def generate_answer(self, query, context_documents):
        context = '\n\n'.join([doc['document'] for doc in context_documents])
        
        prompt = f'''基于以下文档内容回答问题：

文档内容：
{context}

问题：{query}

请基于上述文档内容提供准确的答案。如果文档中没有相关信息，请明确说明。

答案：'''

        if 'gpt' in self.model_name:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的AI助手，擅长基于提供的文档内容回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        else:
            response = self.generator(prompt, max_length=500, num_return_sequences=1)
            return response[0]['generated_text'][len(prompt):]
```

### 2.2 完整RAG系统
```python
class RAGSystem:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', 
                 generation_model_name='gpt-3.5-turbo',
                 openai_api_key=None,
                 chunk_size=512,
                 chunk_overlap=50):
        self.doc_processor = DocumentProcessor(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.generation_model = GenerationModel(generation_model_name)
        self.vector_db = None
        self.openai_api_key = openai_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def build_knowledge_base(self, documents, save_path=None):
        all_chunks = []
        all_metadata = []
        
        for doc_id, document in enumerate(documents):
            chunks, metadata = self.doc_processor.process_document(document)
            
            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'source_file': documents[doc_id] if isinstance(documents[doc_id], str) else f"document_{doc_id}"
                })
                all_metadata.append(chunk_metadata)
        
        # 生成嵌入
        embeddings = self.embedding_model.encode(all_chunks)
        
        # 创建向量数据库
        self.vector_db = VectorDatabase(embeddings.shape[1])
        self.vector_db.add_documents(embeddings, all_chunks, all_metadata)
        
        # 保存系统
        if save_path:
            self.save_system(save_path)
    
    def query(self, question, top_k=5, use_rerank=False, include_structure_info=False):
        if self.vector_db is None:
            raise ValueError("知识库尚未构建，请先调用build_knowledge_base方法")
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode_query(question)
        
        # 检索相关文档
        retrieved_docs = self.vector_db.search(query_embedding, k=top_k)
        
        if not retrieved_docs:
            return {
                'answer': "抱歉，没有找到相关信息来回答您的问题。",
                'retrieved_documents': [],
                'confidence': 0.0
            }
        
        # 处理检索到的文档
        for doc in retrieved_docs:
            doc['similarity_score'] = doc.pop('score')
            doc['content'] = doc.pop('document')
            if include_structure_info:
                doc['document_type'] = 'text'
        
        # 生成答案
        answer = self.generation_model.generate_answer(question, retrieved_docs)
        
        # 计算置信度
        confidence = sum(doc['similarity_score'] for doc in retrieved_docs) / len(retrieved_docs)
        
        return {
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'confidence': confidence,
            'query': question
        }
    
    def search_documents(self, query, top_k=5):
        if self.vector_db is None:
            raise ValueError("知识库尚未构建，请先调用build_knowledge_base方法")
        
        query_embedding = self.embedding_model.encode_query(query)
        retrieved_docs = self.vector_db.search(query_embedding, k=top_k)
        
        results = []
        for doc in retrieved_docs:
            results.append({
                'content': doc['document'],
                'similarity_score': doc['score'],
                'source_file': doc['metadata'].get('source_file', 'unknown')
            })
        
        return results
    
    def save_system(self, path):
        os.makedirs(path, exist_ok=True)
        # 实现保存功能
        pass
    
    def load_system(self, path):
        # 实现加载功能
        pass
    
    def get_system_stats(self):
        stats = {
            'vector_database': {
                'active_documents': len(self.vector_db.documents) if self.vector_db else 0
            },
            'embedding_model': {
                'model_name': self.embedding_model.model.get_sentence_embedding_dimension() if hasattr(self.embedding_model, 'model') else 0,
                'vector_size': self.embedding_model.model.get_sentence_embedding_dimension() if hasattr(self.embedding_model, 'model') else 0
            },
            'llm_model': {
                'model_name': self.generation_model.model_name if hasattr(self.generation_model, 'model_name') else 'unknown'
            },
            'document_processing': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        }
        return stats
```

## 3. 高级RAG技术

### 3.1 混合检索
结合稀疏检索（如BM25）和密集检索（向量相似度）：

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, alpha=0.5):
        self.alpha = alpha  # 密集检索权重
        self.bm25 = None
        self.vector_db = None
        self.documents = []
    
    def build_index(self, documents, embeddings):
        self.documents = documents
        
        # 构建BM25索引
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # 构建向量索引
        self.vector_db = VectorDatabase(embeddings.shape[1])
        self.vector_db.add_documents(embeddings, documents, 
                                   [{'id': i} for i in range(len(documents))])
    
    def search(self, query, k=5):
        # BM25检索
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 向量检索
        query_embedding = self.embedding_model.encode_query(query)
        vector_results = self.vector_db.search(query_embedding, k=len(self.documents))
        vector_scores = np.array([result['score'] for result in vector_results])
        
        # 标准化分数
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min())
        
        # 混合分数
        hybrid_scores = self.alpha * vector_scores + (1 - self.alpha) * bm25_scores
        
        # 排序并返回top-k
        top_indices = np.argsort(hybrid_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': hybrid_scores[idx],
                'bm25_score': bm25_scores[idx],
                'vector_score': vector_scores[idx]
            })
        
        return results
```

### 3.2 重排序机制
使用更强的模型对检索结果进行重新排序：

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, documents, top_k=5):
        # 准备查询-文档对
        pairs = [[query, doc['document']] for doc in documents]
        
        # 计算相关性分数
        scores = self.model.predict(pairs)
        
        # 添加重排序分数
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # 按重排序分数排序
        reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked_docs[:top_k]

# 在RAG系统中集成重排序
class AdvancedRAGSystem(RAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = Reranker()
    
    def query(self, question, top_k=5, use_reranking=True):
        # 初始检索更多文档
        initial_k = top_k * 3 if use_reranking else top_k
        
        query_embedding = self.embedding_model.encode_query(question)
        retrieved_docs = self.vector_db.search(query_embedding, k=initial_k)
        
        # 重排序
        if use_reranking:
            retrieved_docs = self.reranker.rerank(question, retrieved_docs, top_k)
        
        # 生成答案
        answer = self.generation_model.generate_answer(question, retrieved_docs)
        
        return {
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'query': question
        }
```

### 3.3 查询扩展
扩展用户查询以提高检索效果：

```python
class QueryExpander:
    def __init__(self, expansion_model='gpt-3.5-turbo'):
        self.model = expansion_model
        if 'gpt' in expansion_model:
            import openai
            self.client = openai.OpenAI()
    
    def expand_query(self, query):
        prompt = f'''请为以下查询生成3-5个相关的同义词或相关术语，用于改进搜索效果：

原始查询：{query}

相关术语（用逗号分隔）：'''

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个专业的查询扩展助手。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        expanded_terms = response.choices[0].message.content.strip()
        return f"{query} {expanded_terms}"
```

## 4. 评估和优化

### 4.1 评估指标

#### 检索质量指标
```python
def calculate_retrieval_metrics(retrieved_docs, relevant_docs, k=5):
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    # 精确率@k
    precision_at_k = len(retrieved_set & relevant_set) / k
    
    # 召回率@k
    recall_at_k = len(retrieved_set & relevant_set) / len(relevant_set)
    
    # F1@k
    if precision_at_k + recall_at_k > 0:
        f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
    else:
        f1_at_k = 0
    
    return {
        'precision@k': precision_at_k,
        'recall@k': recall_at_k,
        'f1@k': f1_at_k
    }
```

#### 生成质量指标
```python
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

def calculate_generation_metrics(generated_answer, reference_answer):
    # ROUGE分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_answer, generated_answer)
    
    # BLEU分数
    reference_tokens = reference_answer.split()
    generated_tokens = generated_answer.split()
    bleu_score = sentence_bleu([reference_tokens], generated_tokens)
    
    return {
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure,
        'bleu': bleu_score
    }
```

### 4.2 系统优化策略

#### 文档分块优化
```python
class AdaptiveChunker:
    def __init__(self, min_chunk_size=200, max_chunk_size=800):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_by_structure(self, text):
        # 按段落分块
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= self.max_chunk_size:
                current_chunk += paragraph + '\n\n'
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
        
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
```

#### 缓存机制
```python
import hashlib
import pickle
from functools import lru_cache

class RAGCache:
    def __init__(self, cache_size=1000):
        self.cache = {}
        self.cache_size = cache_size
    
    def _hash_query(self, query):
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query):
        query_hash = self._hash_query(query)
        return self.cache.get(query_hash)
    
    def set(self, query, result):
        query_hash = self._hash_query(query)
        
        if len(self.cache) >= self.cache_size:
            # 移除最旧的条目
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[query_hash] = result
    
    def clear(self):
        self.cache.clear()
```

## 5. 部署和监控

### 5.1 系统部署
```python
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
rag_system = None

@app.route('/query', methods=['POST'])
def query_endpoint():
    try:
        data = request.json
        question = data.get('question', '')
        top_k = data.get('top_k', 5)
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        result = rag_system.query(question, top_k=top_k)
        
        return jsonify({
            'answer': result['answer'],
            'sources': [
                {
                    'content': doc['document'][:200] + '...',
                    'score': doc['score']
                }
                for doc in result['retrieved_documents']
            ]
        })
    
    except Exception as e:
        logging.error(f"Query error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # 初始化RAG系统
    rag_system = RAGSystem()
    # 加载知识库...
    
    app.run(host='0.0.0.0', port=5000)
```

### 5.2 性能监控
```python
import time
from collections import defaultdict

class RAGMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def log_query(self, query, response_time, retrieved_count, answer_length):
        self.metrics['response_times'].append(response_time)
        self.metrics['retrieved_counts'].append(retrieved_count)
        self.metrics['answer_lengths'].append(answer_length)
        self.metrics['query_count'] += 1
    
    def get_stats(self):
        if not self.metrics['response_times']:
            return {}
        
        return {
            'avg_response_time': sum(self.metrics['response_times']) / len(self.metrics['response_times']),
            'avg_retrieved_count': sum(self.metrics['retrieved_counts']) / len(self.metrics['retrieved_counts']),
            'avg_answer_length': sum(self.metrics['answer_lengths']) / len(self.metrics['answer_lengths']),
            'total_queries': len(self.metrics['response_times'])
        }

# 在RAG系统中集成监控
class MonitoredRAGSystem(RAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = RAGMonitor()
    
    def query(self, question, top_k=5):
        start_time = time.time()
        
        result = super().query(question, top_k)
        
        response_time = time.time() - start_time
        retrieved_count = len(result['retrieved_documents'])
        answer_length = len(result['answer'])
        
        self.monitor.log_query(question, response_time, retrieved_count, answer_length)
        
        return result
```

这个指南提供了构建高质量RAG系统的完整框架，包括核心组件、高级技术、评估方法和部署策略。通过这些技术，可以构建出准确、高效、可扩展的RAG系统。

"""

    # 创建文档文件
    os.makedirs("enhanced_docs", exist_ok=True)
    
    with open("enhanced_docs/ai_technology_guide.txt", "w", encoding="utf-8") as f:
        f.write(doc1_content)
    
    with open("enhanced_docs/nlp_practical_guide.txt", "w", encoding="utf-8") as f:
        f.write(doc2_content)
    
    with open("enhanced_docs/rag_system_guide.txt", "w", encoding="utf-8") as f:
        f.write(doc3_content)
    
    print("增强示例文档已创建在 enhanced_docs/ 目录中")
    return [
        "enhanced_docs/ai_technology_guide.txt", 
        "enhanced_docs/nlp_practical_guide.txt", 
        "enhanced_docs/rag_system_guide.txt"
    ]

def summarize_text(text, sentences_count=3):
    # 解析文本
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    # 创建摘要器
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    
    # 生成摘要
    summary = summarizer(parser.document, sentences_count)
    
    return [str(sentence) for sentence in summary]

def demo_enhanced_rag_system():
    """演示增强RAG系统的使用"""
    
    print("=== 增强RAG系统演示（GPT-4o + text-embedding-large）===\n")
    
    # 检查API密钥
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("警告: 未设置OPENAI_API_KEY环境变量")
        print("请设置您的OpenAI API密钥以使用GPT-4o和text-embedding-large")
        print("示例: set OPENAI_API_KEY=your_api_key_here")
        print("\n继续演示其他功能...\n")
        return
    
    try:
        # 1. 创建增强示例文档
        print("1. 创建增强示例文档...")
        file_paths = create_enhanced_sample_documents()
        
        # 2. 初始化增强RAG系统
        print("2. 初始化增强RAG系统...")
        rag = RAGSystem(
            embedding_model="text-embedding-large",
            llm_model="gpt-4o",
            openai_api_key=api_key,
            chunk_size=800,
            chunk_overlap=100
        )
        
        # 3. 构建知识库
        print("3. 构建知识库...")
        rag.build_knowledge_base(file_paths, save_path="enhanced_rag")
        
        # 4. 显示系统统计信息
        print("4. 系统统计信息:")
        stats = rag.get_system_stats()
        print(f"   - 文档数量: {stats['vector_database']['active_documents']}")
        print(f"   - 嵌入模型: {stats['embedding_model']['model_name']}")
        print(f"   - 向量维度: {stats['embedding_model']['vector_size']}")
        print(f"   - LLM模型: {stats['llm_model']['model_name']}")
        print(f"   - 分块大小: {stats['document_processing']['chunk_size']}")
        
        # 5. 测试查询
        print("\n5. 测试查询:")
        
        test_queries = [
            "什么是机器学习？请详细解释其主要类型。",
            "深度学习中的CNN和RNN有什么区别？",
            "如何构建一个高质量的RAG系统？",
            "自然语言处理的核心任务有哪些？",
            "Transformer架构的主要优势是什么？",
            "如何评估RAG系统的性能？"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n查询 {i}: {query}")
            print("-" * 80)
            
            result = rag.query(
                query, 
                top_k=3, 
                use_rerank=True,
                include_structure_info=True
            )
            
            print(f"答案: {result['answer'][:300]}...")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"检索到 {len(result['retrieved_documents'])} 个相关文档")
            
            # 显示检索到的文档片段
            for j, doc in enumerate(result['retrieved_documents'][:2]):
                print(f"  文档{j+1} (相似度: {doc['similarity_score']:.3f})")
                print(f"    来源: {doc['source_file']}")
                print(f"    类型: {doc['document_type']}")
                print(f"    内容: {doc['content'][:150]}...")
        
        # 6. 测试文档搜索功能
        print("\n6. 测试文档搜索功能:")
        search_query = "神经网络的基本组件"
        search_results = rag.search_documents(search_query, top_k=3)
        
        print(f"搜索查询: {search_query}")
        for i, result in enumerate(search_results):
            print(f"  结果{i+1} (相似度: {result['similarity_score']:.3f})")
            print(f"    来源: {result['source_file']}")
            print(f"    内容: {result['content'][:100]}...")
        
        # 7. 保存和加载测试
        print("\n7. 测试保存和加载功能...")
        rag.save_system("enhanced_rag")
        
        # 创建新的RAG实例并加载
        new_rag = RAGSystem(
            embedding_model="text-embedding-large",
            llm_model="gpt-4o",
            openai_api_key=api_key
        )
        new_rag.load_system("enhanced_rag")
        
        print("   系统保存和加载成功！")
        
        # 8. 性能测试
        print("\n8. 性能测试:")
        import time
        
        start_time = time.time()
        test_result = new_rag.query("什么是深度学习？", top_k=3)
        end_time = time.time()
        
        print(f"   查询响应时间: {end_time - start_time:.2f} 秒")
        print(f"   答案长度: {len(test_result['answer'])} 字符")
        
        print("\n=== 演示完成 ===")
        print("\n增强RAG系统特性:")
        print("✓ 使用GPT-4o作为生成模型，提供更高质量的答案")
        print("✓ 使用text-embedding-large，提供更精确的语义理解")
        print("✓ 智能文档分块，保持语义完整性")
        print("✓ 增强文本处理，支持多种文档格式")
        print("✓ 文档结构分析，提供更丰富的上下文信息")
        print("✓ 批量嵌入处理，提高系统效率")
        print("✓ 可选的LLM重排序，提高检索精度")
        print("✓ 完整的保存/加载机制")
        
        print("\n使用说明:")
        print("1. 将您的.txt文档放在指定目录")
        print("2. 调用 rag.build_knowledge_base(['file1.txt', 'file2.txt']) 构建知识库")
        print("3. 调用 rag.query('your question') 进行查询")
        print("4. 使用 rag.save_system('path') 保存系统")
        print("5. 使用 rag.load_system('path') 加载系统")
        print("6. 使用 rag.search_documents('query') 仅搜索文档")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_enhanced_rag_system()