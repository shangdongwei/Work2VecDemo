import os
import re
import nltk
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pickle
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Word2VecEmbedding:
    def __init__(self, vector_size=300, window=5, min_count=2, workers=4, epochs=100):
        """
        初始化Word2Vec嵌入模型
        
        Args:
            vector_size: 词向量维度
            window: 上下文窗口大小
            min_count: 最小词频
            workers: 训练线程数
            epochs: 训练轮数
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None
        self.stop_words = set(ENGLISH_STOP_WORDS)
        
        # 下载NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def preprocess_text(self, text):
        """
        文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的词列表
        """
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符，只保留字母和空格
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 使用gensim的simple_preprocess进行分词
        tokens = simple_preprocess(text, deacc=True)
        
        # 移除停用词和长度小于3的词
        tokens = [token for token in tokens if token not in self.stop_words and len(token) >= 3]
        
        return tokens
    
    def load_and_preprocess_documents(self, file_paths):
        """
        加载并预处理文档
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            预处理后的文档列表
        """
        documents = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                # 按句子分割
                sentences = nltk.sent_tokenize(content)
                
                for sentence in sentences:
                    tokens = self.preprocess_text(sentence)
                    if len(tokens) >= 3:  # 只保留至少有3个词的句子
                        documents.append(tokens)
                        
                logger.info(f"已处理文件: {file_path}")
                
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {e}")
        
        logger.info(f"总共处理了 {len(documents)} 个句子")
        return documents
    
    def train_model(self, documents):
        """
        训练Word2Vec模型
        
        Args:
            documents: 预处理后的文档列表
        """
        logger.info("开始训练Word2Vec模型...")
        
        self.model = Word2Vec(
            sentences=documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=1,  # 使用Skip-gram
            hs=0,  # 使用负采样
            negative=10,  # 负采样数量
            alpha=0.025,  # 初始学习率
            min_alpha=0.0001  # 最小学习率
        )
        
        logger.info(f"模型训练完成！词汇表大小: {len(self.model.wv.key_to_index)}")
    
    def get_sentence_embedding(self, sentence):
        """
        获取句子的嵌入向量（通过平均词向量）
        
        Args:
            sentence: 输入句子
            
        Returns:
            句子的嵌入向量
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        tokens = self.preprocess_text(sentence)
        vectors = []
        
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            # 如果没有找到任何词，返回零向量
            return np.zeros(self.vector_size)
    
    def get_word_embedding(self, word):
        """
        获取单词的嵌入向量
        
        Args:
            word: 输入单词
            
        Returns:
            单词的嵌入向量
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        word = word.lower()
        if word in self.model.wv:
            return self.model.wv[word]
        else:
            return np.zeros(self.vector_size)
    
    def find_similar_words(self, word, topn=10):
        """
        查找相似词
        
        Args:
            word: 输入单词
            topn: 返回相似词数量
            
        Returns:
            相似词列表
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        word = word.lower()
        if word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        else:
            return []
    
    def save_model(self, model_path):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        self.model.save(model_path)
        logger.info(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型路径
        """
        self.model = Word2Vec.load(model_path)
        logger.info(f"模型已从 {model_path} 加载")

# 示例使用
if __name__ == "__main__":
    # 创建Word2Vec嵌入模型实例
    w2v_embedding = Word2VecEmbedding(vector_size=300, epochs=100)
    
    # 示例：如果有文本文件
    # file_paths = ["sample_text.txt"]  # 替换为您的文本文件路径
    # documents = w2v_embedding.load_and_preprocess_documents(file_paths)
    # w2v_embedding.train_model(documents)
    # w2v_embedding.save_model("word2vec_model.model")
    
    print("Word2Vec嵌入模型类已创建完成！")