import os
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import json

from word2vec_model import Word2VecEmbedding
from vector_database import FaissVectorDatabase
from qwen_llm import QwenLLM

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, 
                 vector_size: int = 300,
                 index_type: str = "flat",
                 qwen_model: str = "qwen-turbo",
                 api_key: str = None):
        """
        初始化RAG系统
        
        Args:
            vector_size: 词向量维度
            index_type: Faiss索引类型
            qwen_model: Qwen模型名称
            api_key: DashScope API密钥
        """
        self.vector_size = vector_size
        
        # 初始化组件
        self.embedding_model = Word2VecEmbedding(vector_size=vector_size)
        self.vector_db = FaissVectorDatabase(dimension=vector_size, index_type=index_type)
        self.llm = QwenLLM(api_key=api_key, model_name=qwen_model)
        
        # 文档处理参数
        self.chunk_size = 500  # 文档分块大小
        self.chunk_overlap = 50  # 分块重叠大小
        
        logger.info("RAG系统初始化完成")
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        将长文本分割成块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def load_documents_from_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        从文件加载文档
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            文档信息列表
        """
        documents = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # 分割文档为块
                chunks = self.split_text_into_chunks(content)
                
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) > 50:  # 只保留有意义的块
                        doc_info = {
                            'content': chunk,
                            'source_file': file_path,
                            'chunk_id': i,
                            'metadata': {
                                'file_name': os.path.basename(file_path),
                                'file_path': file_path,
                                'chunk_index': i,
                                'total_chunks': len(chunks)
                            }
                        }
                        documents.append(doc_info)
                
                logger.info(f"已加载文件: {file_path}, 分割为 {len(chunks)} 个块")
                
            except Exception as e:
                logger.error(f"加载文件 {file_path} 时出错: {e}")
        
        logger.info(f"总共加载了 {len(documents)} 个文档块")
        return documents
    
    def build_knowledge_base(self, file_paths: List[str], save_path: str = None):
        """
        构建知识库
        
        Args:
            file_paths: 文档文件路径列表
            save_path: 保存路径前缀
        """
        logger.info("开始构建知识库...")
        
        # 1. 加载和预处理文档
        documents = self.load_documents_from_files(file_paths)
        
        if not documents:
            raise ValueError("没有成功加载任何文档")
        
        # 2. 训练Word2Vec模型
        logger.info("训练Word2Vec嵌入模型...")
        all_texts = [doc['content'] for doc in documents]
        
        # 为Word2Vec准备训练数据
        training_sentences = []
        for text in all_texts:
            sentences = self.embedding_model.preprocess_text(text)
            if sentences:
                training_sentences.append(sentences)
        
        # 添加来自所有文件的句子用于训练
        all_file_sentences = self.embedding_model.load_and_preprocess_documents(file_paths)
        training_sentences.extend(all_file_sentences)
        
        self.embedding_model.train_model(training_sentences)
        
        # 3. 生成文档嵌入
        logger.info("生成文档嵌入向量...")
        embeddings = []
        valid_documents = []
        
        for doc in tqdm(documents, desc="生成嵌入"):
            embedding = self.embedding_model.get_sentence_embedding(doc['content'])
            
            # 检查嵌入是否有效（不全为零）
            if np.any(embedding):
                embeddings.append(embedding)
                valid_documents.append(doc)
        
        if not embeddings:
            raise ValueError("没有生成有效的文档嵌入")
        
        embeddings = np.array(embeddings)
        
        # 4. 构建向量数据库
        logger.info("构建向量数据库...")
        doc_texts = [doc['content'] for doc in valid_documents]
        doc_metadata = [doc['metadata'] for doc in valid_documents]
        
        self.vector_db.add_documents(doc_texts, embeddings, doc_metadata)
        
        # 5. 保存模型和数据库
        if save_path:
            self.save_system(save_path)
        
        logger.info(f"知识库构建完成！包含 {len(valid_documents)} 个文档块")
    
    def query(self, 
              question: str, 
              top_k: int = 5, 
              similarity_threshold: float = 0.3,
              use_rerank: bool = True) -> Dict[str, Any]:
        """
        查询RAG系统
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            similarity_threshold: 相似度阈值
            use_rerank: 是否使用LLM重新排序
            
        Returns:
            包含答案和相关信息的字典
        """
        logger.info(f"处理查询: {question}")
        
        # 1. 生成查询嵌入
        query_embedding = self.embedding_model.get_sentence_embedding(question)
        
        if not np.any(query_embedding):
            return {
                'answer': '抱歉，无法理解您的问题，请尝试重新表述。',
                'retrieved_documents': [],
                'confidence': 0.0
            }
        
        # 2. 检索相关文档
        search_results = self.vector_db.search(
            query_embedding, 
            k=top_k * 2,  # 检索更多文档用于重排序
            threshold=similarity_threshold
        )
        
        if not search_results:
            return {
                'answer': '抱歉，在知识库中没有找到相关信息来回答您的问题。',
                'retrieved_documents': [],
                'confidence': 0.0
            }
        
        # 3. 可选的LLM重新排序
        if use_rerank and len(search_results) > top_k:
            logger.info("使用LLM重新排序检索结果...")
            reranked_results = self._rerank_documents(question, search_results[:top_k*2])
            search_results = reranked_results[:top_k]
        else:
            search_results = search_results[:top_k]
        
        # 4. 生成答案
        retrieved_docs = [result[0] for result in search_results]
        answer = self.llm.generate_rag_response(question, retrieved_docs)
        
        # 5. 计算置信度
        avg_similarity = np.mean([result[1] for result in search_results])
        
        return {
            'answer': answer,
            'retrieved_documents': [
                {
                    'content': result[0],
                    'similarity_score': result[1],
                    'metadata': result[2]
                }
                for result in search_results
            ],
            'confidence': float(avg_similarity),
            'query': question
        }
    
    def _rerank_documents(self, query: str, search_results: List[Tuple]) -> List[Tuple]:
        """
        使用LLM重新排序文档
        
        Args:
            query: 查询文本
            search_results: 搜索结果列表
            
        Returns:
            重新排序后的结果
        """
        scored_results = []
        
        for doc, similarity, metadata in search_results:
            # 使用LLM评估相关性
            relevance_score = self.llm.check_relevance(query, doc)
            
            # 结合向量相似度和LLM相关性评分
            combined_score = 0.6 * similarity + 0.4 * relevance_score
            
            scored_results.append((doc, combined_score, metadata))
        
        # 按组合分数排序
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None):
        """
        添加单个文档到知识库
        
        Args:
            content: 文档内容
            metadata: 文档元数据
        """
        # 分割文档
        chunks = self.split_text_into_chunks(content)
        
        embeddings = []
        doc_texts = []
        doc_metadata = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:
                embedding = self.embedding_model.get_sentence_embedding(chunk)
                
                if np.any(embedding):
                    embeddings.append(embedding)
                    doc_texts.append(chunk)
                    
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update({
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    })
                    doc_metadata.append(chunk_metadata)
        
        if embeddings:
            embeddings = np.array(embeddings)
            self.vector_db.add_documents(doc_texts, embeddings, doc_metadata)
            logger.info(f"添加了 {len(embeddings)} 个文档块到知识库")
    
    def save_system(self, save_path: str):
        """
        保存RAG系统
        
        Args:
            save_path: 保存路径前缀
        """
        # 保存Word2Vec模型
        self.embedding_model.save_model(f"{save_path}_word2vec.model")
        
        # 保存向量数据库
        self.vector_db.save_database(f"{save_path}_vectordb")
        
        # 保存系统配置
        config = {
            'vector_size': self.vector_size,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        with open(f"{save_path}_config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"RAG系统已保存到: {save_path}")
    
    def load_system(self, load_path: str):
        """
        加载RAG系统
        
        Args:
            load_path: 加载路径前缀
        """
        # 加载配置
        with open(f"{load_path}_config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.vector_size = config['vector_size']
        self.chunk_size = config['chunk_size']
        self.chunk_overlap = config['chunk_overlap']
        
        # 加载Word2Vec模型
        self.embedding_model.load_model(f"{load_path}_word2vec.model")
        
        # 加载向量数据库
        self.vector_db.load_database(f"{load_path}_vectordb")
        
        logger.info(f"RAG系统已从 {load_path} 加载")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        db_stats = self.vector_db.get_statistics()
        
        return {
            'vector_database': db_stats,
            'embedding_model': {
                'vector_size': self.vector_size,
                'vocabulary_size': len(self.embedding_model.model.wv.key_to_index) if self.embedding_model.model else 0
            },
            'document_processing': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        }

# 示例使用
if __name__ == "__main__":
    # 创建RAG系统实例
    rag = RAGSystem(
        vector_size=300,
        index_type="flat",
        qwen_model="qwen-turbo"
    )
    
    print("RAG系统已创建完成！")
    print("\n使用示例:")
    print("1. 准备英文文本文件")
    print("2. 调用 rag.build_knowledge_base(['file1.txt', 'file2.txt']) 构建知识库")
    print("3. 调用 rag.query('your question') 进行查询")
    print("\n注意：使用前请确保设置了DASHSCOPE_API_KEY环境变量")