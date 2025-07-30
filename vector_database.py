import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FaissVectorDatabase:
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        初始化Faiss向量数据库
        
        Args:
            dimension: 向量维度
            index_type: 索引类型 ("flat", "ivf", "hnsw")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents = []  # 存储原始文档
        self.document_embeddings = []  # 存储文档嵌入
        self.metadata = []  # 存储文档元数据
        
        self._create_index()
    
    def _create_index(self):
        """创建Faiss索引"""
        if self.index_type == "flat":
            # L2距离的平面索引，精确搜索
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            # IVF索引，适合大规模数据
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = 100  # 聚类中心数量
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "hnsw":
            # HNSW索引，快速近似搜索
            M = 32  # 连接数
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        logger.info(f"创建了 {self.index_type} 类型的Faiss索引，维度: {self.dimension}")
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray, metadata: List[Dict] = None):
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档文本列表
            embeddings: 文档嵌入向量数组
            metadata: 文档元数据列表
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"嵌入维度 {embeddings.shape[1]} 与索引维度 {self.dimension} 不匹配")
        
        # 确保embeddings是float32类型
        embeddings = embeddings.astype(np.float32)
        
        # 如果是IVF索引，需要先训练
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("训练IVF索引...")
            self.index.train(embeddings)
        
        # 添加向量到索引
        start_id = len(self.documents)
        self.index.add(embeddings)
        
        # 存储文档和元数据
        self.documents.extend(documents)
        self.document_embeddings.extend(embeddings.tolist())
        
        if metadata is None:
            metadata = [{"id": start_id + i} for i in range(len(documents))]
        self.metadata.extend(metadata)
        
        logger.info(f"添加了 {len(documents)} 个文档到向量数据库")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = None) -> List[Tuple[str, float, Dict]]:
        """
        搜索相似文档
        
        Args:
            query_embedding: 查询向量
            k: 返回结果数量
            threshold: 距离阈值，超过此值的结果将被过滤
            
        Returns:
            (文档, 相似度分数, 元数据) 的列表
        """
        if len(self.documents) == 0:
            return []
        
        # 确保查询向量是正确的形状和类型
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        
        # 执行搜索
        distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Faiss返回-1表示无效结果
                continue
            
            if threshold is not None and distance > threshold:
                continue
            
            # 将L2距离转换为相似度分数 (0-1之间，1表示最相似)
            similarity_score = 1.0 / (1.0 + distance)
            
            results.append((
                self.documents[idx],
                similarity_score,
                self.metadata[idx]
            ))
        
        return results
    
    def delete_document(self, doc_id: int):
        """
        删除文档（注意：Faiss不支持直接删除，这里只是标记）
        
        Args:
            doc_id: 文档ID
        """
        if 0 <= doc_id < len(self.documents):
            self.metadata[doc_id]["deleted"] = True
            logger.info(f"标记文档 {doc_id} 为已删除")
        else:
            raise ValueError(f"无效的文档ID: {doc_id}")
    
    def get_document_count(self) -> int:
        """获取文档数量"""
        return len([doc for i, doc in enumerate(self.documents) 
                   if not self.metadata[i].get("deleted", False)])
    
    def save_database(self, db_path: str):
        """
        保存向量数据库
        
        Args:
            db_path: 数据库保存路径（不包含扩展名）
        """
        # 保存Faiss索引
        faiss.write_index(self.index, f"{db_path}.index")
        
        # 保存文档和元数据
        with open(f"{db_path}.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "document_embeddings": self.document_embeddings,
                "metadata": self.metadata,
                "dimension": self.dimension,
                "index_type": self.index_type
            }, f)
        
        logger.info(f"向量数据库已保存到: {db_path}")
    
    def load_database(self, db_path: str):
        """
        加载向量数据库
        
        Args:
            db_path: 数据库路径（不包含扩展名）
        """
        # 加载Faiss索引
        self.index = faiss.read_index(f"{db_path}.index")
        
        # 加载文档和元数据
        with open(f"{db_path}.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.document_embeddings = data["document_embeddings"]
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]
            self.index_type = data["index_type"]
        
        logger.info(f"向量数据库已从 {db_path} 加载")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        active_docs = [doc for i, doc in enumerate(self.documents) 
                      if not self.metadata[i].get("deleted", False)]
        
        return {
            "total_documents": len(self.documents),
            "active_documents": len(active_docs),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "index_size": self.index.ntotal if self.index else 0
        }

# 示例使用
if __name__ == "__main__":
    # 创建向量数据库
    db = FaissVectorDatabase(dimension=300, index_type="flat")
    
    # 示例数据
    sample_docs = [
        "This is a sample document about machine learning.",
        "Natural language processing is a fascinating field.",
        "Vector databases are useful for similarity search."
    ]
    
    # 创建示例嵌入（实际使用中应该用Word2Vec生成）
    sample_embeddings = np.random.rand(3, 300).astype(np.float32)
    
    # 添加文档
    db.add_documents(sample_docs, sample_embeddings)
    
    # 搜索示例
    query_embedding = np.random.rand(1, 300).astype(np.float32)
    results = db.search(query_embedding, k=2)
    
    print("搜索结果:")
    for doc, score, metadata in results:
        print(f"文档: {doc[:50]}...")
        print(f"相似度: {score:.4f}")
        print(f"元数据: {metadata}")
        print("-" * 50)
    
    print("Faiss向量数据库类已创建完成！")