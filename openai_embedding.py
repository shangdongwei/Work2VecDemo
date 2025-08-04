import openai
import os
import numpy as np
import logging
from typing import List, Dict, Any
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIEmbedding:
    def __init__(self, api_key: str = None, model_name: str = "text-embedding-large"):
        """
        初始化OpenAI嵌入模型
        
        Args:
            api_key: OpenAI API密钥
            model_name: 嵌入模型名称 (text-embedding-large, text-embedding-small等)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("请设置OPENAI_API_KEY环境变量或传入api_key参数")
        
        # 设置OpenAI客户端
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # 获取模型维度信息
        self.vector_size = self._get_model_dimensions()
        self.stop_words = set(ENGLISH_STOP_WORDS)
        
        # 下载NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        logger.info(f"初始化OpenAI嵌入模型: {self.model_name}, 向量维度: {self.vector_size}")
    
    def _get_model_dimensions(self) -> int:
        """
        获取模型的向量维度
        
        Returns:
            向量维度
        """
        # 不同模型的维度映射
        model_dimensions = {
            "text-embedding-large": 3072,
            "text-embedding-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
        
        return model_dimensions.get(self.model_name, 1536)  # 默认1536维
    
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理（对于OpenAI embedding，保持相对简单的预处理）
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 移除特殊字符，但保留基本标点
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text
    
    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            文本的嵌入向量
        """
        try:
            # 预处理文本
            processed_text = self.preprocess_text(text)
            
            if not processed_text.strip():
                return np.zeros(self.vector_size)
            
            # 调用OpenAI API获取嵌入
            response = self.client.embeddings.create(
                model=self.model_name,
                input=processed_text
            )
            
            # 提取嵌入向量
            embedding = np.array(response.data[0].embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"获取嵌入向量时出错: {e}")
            return np.zeros(self.vector_size)
    
    def get_batch_embeddings(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        批量获取文本嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            嵌入向量列表
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # 预处理批量文本
                processed_texts = [self.preprocess_text(text) for text in batch_texts]
                
                # 过滤空文本
                valid_texts = [text for text in processed_texts if text.strip()]
                
                if not valid_texts:
                    # 如果批次中没有有效文本，添加零向量
                    embeddings.extend([np.zeros(self.vector_size)] * len(batch_texts))
                    continue
                
                # 调用OpenAI API获取批量嵌入
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=valid_texts
                )
                
                # 提取嵌入向量
                batch_embeddings = [np.array(data.embedding) for data in response.data]
                
                # 处理原始批次中的空文本
                valid_idx = 0
                for text in processed_texts:
                    if text.strip():
                        embeddings.append(batch_embeddings[valid_idx])
                        valid_idx += 1
                    else:
                        embeddings.append(np.zeros(self.vector_size))
                
                logger.info(f"已处理 {i + len(batch_texts)}/{len(texts)} 个文本")
                
            except Exception as e:
                logger.error(f"批量获取嵌入向量时出错: {e}")
                # 添加零向量作为fallback
                embeddings.extend([np.zeros(self.vector_size)] * len(batch_texts))
        
        return embeddings
    
    def load_and_preprocess_documents(self, file_paths: List[str]) -> List[str]:
        """
        加载并预处理文档（为了兼容性保留此方法）
        
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
                    processed_sentence = self.preprocess_text(sentence)
                    if len(processed_sentence.strip()) > 20:  # 只保留有意义的句子
                        documents.append(processed_sentence)
                        
                logger.info(f"已处理文件: {file_path}")
                
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {e}")
        
        logger.info(f"总共处理了 {len(documents)} 个句子")
        return documents
    
    def train_model(self, documents: List[str]):
        """
        训练模型（OpenAI embedding不需要训练，此方法为兼容性保留）
        
        Args:
            documents: 文档列表
        """
        logger.info("OpenAI嵌入模型无需训练，跳过训练步骤")
        pass
    
    def save_model(self, model_path: str):
        """
        保存模型（OpenAI embedding不需要保存，此方法为兼容性保留）
        
        Args:
            model_path: 模型保存路径
        """
        logger.info("OpenAI嵌入模型无需保存，跳过保存步骤")
        pass
    
    def load_model(self, model_path: str):
        """
        加载模型（OpenAI embedding不需要加载，此方法为兼容性保留）
        
        Args:
            model_path: 模型路径
        """
        logger.info("OpenAI嵌入模型无需加载，跳过加载步骤")
        pass
    
    def find_similar_words(self, word: str, topn: int = 10) -> List[tuple]:
        """
        查找相似词（OpenAI embedding不直接支持，返回空列表）
        
        Args:
            word: 输入单词
            topn: 返回相似词数量
            
        Returns:
            空列表（兼容性方法）
        """
        logger.info("OpenAI嵌入模型不支持直接查找相似词")
        return []
    
    def get_word_embedding(self, word: str) -> np.ndarray:
        """
        获取单词的嵌入向量
        
        Args:
            word: 输入单词
            
        Returns:
            单词的嵌入向量
        """
        return self.get_sentence_embedding(word)

# 示例使用
if __name__ == "__main__":
    try:
        print("OpenAI嵌入模型类已创建完成！")
        print("使用前请确保设置了OPENAI_API_KEY环境变量")
    except Exception as e:
        print(f"初始化OpenAI嵌入模型时出错: {e}")
        print("请确保设置了正确的OPENAI_API_KEY环境变量")