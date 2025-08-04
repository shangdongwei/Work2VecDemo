import os
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import json

from openai_embedding import OpenAIEmbedding
from vector_database import FaissVectorDatabase
from gpt4o_llm import GPT4oLLM
from enhanced_text_processor import EnhancedTextProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, 
                 embedding_model: str = "text-embedding-large",
                 index_type: str = "flat",
                 llm_model: str = "gpt-4o",
                 openai_api_key: str = None,
                 chunk_size: int = 800,
                 chunk_overlap: int = 100):
        """
        初始化RAG系统
        
        Args:
            embedding_model: OpenAI嵌入模型名称
            index_type: Faiss索引类型
            llm_model: GPT模型名称
            openai_api_key: OpenAI API密钥
            chunk_size: 文档分块大小
            chunk_overlap: 分块重叠大小
        """
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化组件
        self.embedding_model = OpenAIEmbedding(
            api_key=openai_api_key, 
            model_name=embedding_model
        )
        
        self.vector_db = FaissVectorDatabase(
            dimension=self.embedding_model.vector_size, 
            index_type=index_type
        )
        
        self.llm = GPT4oLLM(
            api_key=openai_api_key, 
            model_name=llm_model
        )
        
        self.text_processor = EnhancedTextProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info("RAG系统初始化完成")
        logger.info(f"嵌入模型: {embedding_model}, 向量维度: {self.embedding_model.vector_size}")
        logger.info(f"LLM模型: {llm_model}")
    
    def load_documents_from_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        从文件加载文档
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            文档信息列表
        """
        logger.info("开始加载和处理文档...")
        
        # 使用增强文本处理器加载文档
        document_infos = self.text_processor.load_txt_files(file_paths)
        
        documents = []
        
        for doc_info in document_infos:
            if not doc_info:
                continue
            
            # 为每个文档块创建文档条目
            for i, chunk in enumerate(doc_info['chunks']):
                if len(chunk.strip()) > 50:  # 只保留有意义的块
                    # 提取文档结构信息
                    structure_info = self.text_processor.extract_document_structure(chunk)
                    
                    doc_entry = {
                        'content': chunk,
                        'source_file': doc_info['file_path'],
                        'chunk_id': i,
                        'metadata': {
                            **doc_info['metadata'],
                            'chunk_index': i,
                            'total_chunks': doc_info['chunk_count'],
                            'chunk_length': len(chunk),
                            'structure': structure_info
                        }
                    }
                    documents.append(doc_entry)
        
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
        
        # 2. 生成文档嵌入
        logger.info("生成文档嵌入向量...")
        doc_texts = [doc['content'] for doc in documents]
        
        # 使用批量处理提高效率
        embeddings = self.embedding_model.get_batch_embeddings(doc_texts, batch_size=50)
        
        # 3. 过滤有效嵌入
        valid_documents = []
        valid_embeddings = []
        
        for doc, embedding in zip(documents, embeddings):
            # 检查嵌入是否有效（不全为零）
            if np.any(embedding) and not np.isnan(embedding).any():
                valid_documents.append(doc)
                valid_embeddings.append(embedding)
        
        if not valid_embeddings:
            raise ValueError("没有生成有效的文档嵌入")
        
        valid_embeddings = np.array(valid_embeddings)
        
        # 4. 构建向量数据库
        logger.info("构建向量数据库...")
        doc_texts = [doc['content'] for doc in valid_documents]
        doc_metadata = [doc['metadata'] for doc in valid_documents]
        
        self.vector_db.add_documents(doc_texts, valid_embeddings, doc_metadata)
        
        # 5. 保存模型和数据库
        if save_path:
            self.save_system(save_path)
        
        logger.info(f"知识库构建完成！包含 {len(valid_documents)} 个文档块")
    
    def query(self, 
              question: str, 
              top_k: int = 5, 
              similarity_threshold: float = 0.3,
              use_rerank: bool = True,
              include_structure_info: bool = True) -> Dict[str, Any]:
        """
        查询RAG系统
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            similarity_threshold: 相似度阈值
            use_rerank: 是否使用LLM重新排序
            include_structure_info: 是否包含文档结构信息
            
        Returns:
            包含答案和相关信息的字典
        """
        logger.info(f"处理查询: {question}")
        
        # 1. 生成查询嵌入
        query_embedding = self.embedding_model.get_sentence_embedding(question)
        
        if not np.any(query_embedding) or np.isnan(query_embedding).any():
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
        
        # 4. 准备上下文文档
        retrieved_docs = []
        context_texts = []
        
        for result in search_results:
            doc_content, similarity_score, metadata = result
            
            # 构建增强的上下文
            if include_structure_info and 'structure' in metadata:
                structure = metadata['structure']
                context_prefix = f"[文档类型: {metadata.get('document_type', '未知')}]"
                if structure.get('headings'):
                    context_prefix += f" [标题: {structure['headings'][0] if structure['headings'] else '无'}]"
                enhanced_content = f"{context_prefix}\n{doc_content}"
            else:
                enhanced_content = doc_content
            
            context_texts.append(enhanced_content)
            
            retrieved_docs.append({
                'content': doc_content,
                'similarity_score': float(similarity_score),
                'metadata': metadata,
                'source_file': metadata.get('file_name', '未知'),
                'document_type': metadata.get('document_type', '未知')
            })
        
        # 5. 生成答案
        answer = self.llm.generate_rag_response(question, context_texts)
        
        # 6. 计算置信度
        avg_similarity = np.mean([result[1] for result in search_results])
        
        return {
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'confidence': float(avg_similarity),
            'query': question,
            'total_retrieved': len(search_results)
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
            
            # 结合向量相似度和LLM相关性评分，并考虑文档类型
            doc_type_bonus = 0.1 if metadata.get('document_type') in ['reference', 'guide'] else 0
            combined_score = 0.6 * similarity + 0.3 * relevance_score + doc_type_bonus
            
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
        # 使用文本处理器处理内容
        cleaned_content = self.text_processor.clean_text(content)
        chunks = self.text_processor.intelligent_chunking(cleaned_content)
        
        embeddings = []
        doc_texts = []
        doc_metadata = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:
                embedding = self.embedding_model.get_sentence_embedding(chunk)
                
                if np.any(embedding) and not np.isnan(embedding).any():
                    embeddings.append(embedding)
                    doc_texts.append(chunk)
                    
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update({
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'structure': self.text_processor.extract_document_structure(chunk)
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
        # 保存向量数据库
        self.vector_db.save_database(f"{save_path}_vectordb")
        
        # 保存系统配置
        config = {
            'embedding_model': self.embedding_model_name,
            'llm_model': self.llm_model_name,
            'vector_size': self.embedding_model.vector_size,
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
        
        self.embedding_model_name = config['embedding_model']
        self.llm_model_name = config['llm_model']
        self.chunk_size = config['chunk_size']
        self.chunk_overlap = config['chunk_overlap']
        
        # 加载向量数据库
        self.vector_db.load_database(f"{load_path}_vectordb")
        
        logger.info(f"RAG系统已从 {load_path} 加载")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        db_stats = self.vector_db.get_statistics()
        
        return {
            'vector_database': db_stats,
            'embedding_model': {
                'model_name': self.embedding_model_name,
                'vector_size': self.embedding_model.vector_size
            },
            'llm_model': {
                'model_name': self.llm_model_name
            },
            'document_processing': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        }
    
    def search_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        仅搜索文档，不生成答案
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            
        Returns:
            搜索结果列表
        """
        query_embedding = self.embedding_model.get_sentence_embedding(query)
        
        if not np.any(query_embedding) or np.isnan(query_embedding).any():
            return []
        
        search_results = self.vector_db.search(query_embedding, k=top_k)
        
        results = []
        for doc_content, similarity_score, metadata in search_results:
            results.append({
                'content': doc_content,
                'similarity_score': float(similarity_score),
                'metadata': metadata,
                'source_file': metadata.get('file_name', '未知')
            })
        
        return results

# 示例使用
if __name__ == "__main__":
    # 创建RAG系统实例
    rag = RAGSystem(
        embedding_model="text-embedding-large",
        llm_model="gpt-4o",
        chunk_size=800,
        chunk_overlap=100
    )
    
    print("增强RAG系统已创建完成！")
    print("\n使用示例:")
    print("1. 准备txt文本文件")
    print("2. 调用 rag.build_knowledge_base(['file1.txt', 'file2.txt']) 构建知识库")
    print("3. 调用 rag.query('your question') 进行查询")
    print("\n注意：使用前请确保设置了OPENAI_API_KEY环境变量")