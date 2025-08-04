    return [str(sentence) for sentence in summary]

# 示例使用
long_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language. The goal of NLP is to enable computers to understand, interpret 
and manipulate human language. This involves several challenges including speech 
recognition, natural language understanding, and natural language generation.
"""

summary = summarize_text(long_text, 2)
print("Summary:")
for sentence in summary:
    print(f"- {sentence}")

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
                 generation_model_name='gpt-3.5-turbo'):
        self.doc_processor = DocumentProcessor()
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.generation_model = GenerationModel(generation_model_name)
        self.vector_db = None
    
    def build_knowledge_base(self, documents):
        all_chunks = []
        all_metadata = []
        
        for doc_id, document in enumerate(documents):
            chunks, metadata = self.doc_processor.process_document(document)
            
            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'doc_id': doc_id,
                    'chunk_id': chunk_id
                })
                all_metadata.append(chunk_metadata)
        
        # 生成嵌入
        embeddings = self.embedding_model.encode(all_chunks)
        
        # 创建向量数据库
        self.vector_db = VectorDatabase(embeddings.shape[1])
        self.vector_db.add_documents(embeddings, all_chunks, all_metadata)
    
    def query(self, question, top_k=5):
        if self.vector_db is None:
            raise ValueError("知识库尚未构建，请先调用build_knowledge_base方法")
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode_query(question)
        
        # 检索相关文档
        retrieved_docs = self.vector_db.search(query_embedding, k=top_k)
        
        if not retrieved_docs:
            return "抱歉，没有找到相关信息来回答您的问题。"
        
        # 生成答案
        answer = self.generation_model.generate_answer(question, retrieved_docs)
        
        return {
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'confidence': sum(doc['score'] for doc in retrieved_docs) / len(retrieved_docs)
        }
```

## 3. 高级技术

### 3.1 混合检索
结合稠密检索和稀疏检索：

```python
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, alpha=0.5):
        self.alpha = alpha  # 稠密检索权重
        self.embedding_model = None
        self.bm25 = None
        self.documents = []
    
    def build_index(self, documents, embeddings):
        self.documents = documents
        
        # 构建BM25索引
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # 构建稠密向量索引
        self.vector_db = VectorDatabase(embeddings.shape[1])
        self.vector_db.add_documents(embeddings, documents, 
                                   [{'doc_id': i} for i in range(len(documents))])
    
    def search(self, query, k=5):
        # BM25检索
        bm25_scores = self.bm25.get_scores(query.split())
        
        # 稠密检索
        query_embedding = self.embedding_model.encode_query(query)
        dense_results = self.vector_db.search(query_embedding, k=len(self.documents))
        
        # 混合评分
        final_scores = {}
        for i, score in enumerate(bm25_scores):
            final_scores[i] = (1 - self.alpha) * score
        
        for result in dense_results:
            doc_id = result['metadata']['doc_id']
            if doc_id in final_scores:
                final_scores[doc_id] += self.alpha * result['score']
            else:
                final_scores[doc_id] = self.alpha * result['score']
        
        # 排序并返回top-k
        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_id, score in sorted_docs:
            if doc_id < len(self.documents):
                results.append({
                    'document': self.documents[doc_id],
                    'score': score,
                    'doc_id': doc_id
                })
        
        return results
```

### 3.2 重排序
使用交叉编码器进行重排序：

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
        
        # 重新排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [{'document': doc, 'score': score} for doc, score in scored_docs[:top_k]]
```

### 3.3 增强RAG系统
集成高级技术的RAG系统：

```python
class AdvancedRAGSystem(RAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()
    
    def query(self, question, top_k=5, use_reranking=True):
        # 混合检索
        retrieved_docs = self.hybrid_retriever.search(question, k=top_k*2)
        
        if not retrieved_docs:
            return "抱歉，没有找到相关信息。"
        
        # 重排序
        if use_reranking:
            retrieved_docs = self.reranker.rerank(question, retrieved_docs, top_k)
        
        # 生成答案
        answer = self.generation_model.generate_answer(question, retrieved_docs)
        
        return {
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'confidence': sum(doc['score'] for doc in retrieved_docs) / len(retrieved_docs)
        }
```

## 4. 查询扩展

### 4.1 基于LLM的查询扩展
```python
class QueryExpander:
    def __init__(self, expansion_model='gpt-3.5-turbo'):
        self.expansion_model = expansion_model
        if 'gpt' in expansion_model:
            import openai
            self.client = openai.OpenAI()
    
    def expand_query(self, query):
        prompt = f'''请为以下查询生成3-5个相关的同义词查询，用于改进信息检索效果：

原始查询：{query}

扩展查询（每行一个）：'''

        if 'gpt' in self.expansion_model:
            response = self.client.chat.completions.create(
                model=self.expansion_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的查询扩展助手。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            expanded_queries = response.choices[0].message.content.strip().split('\n')
            return [query] + [q.strip() for q in expanded_queries if q.strip()]
        
        return [query]
```

## 5. 评估指标

### 5.1 检索评估
```python
def calculate_retrieval_metrics(retrieved_docs, relevant_docs, k=5):
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    # 精确率@K
    precision_at_k = len(retrieved_set & relevant_set) / len(retrieved_set) if retrieved_set else 0
    
    # 召回率@K
    recall_at_k = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0
    
    # F1@K
    f1_at_k = 0
    if precision_at_k + recall_at_k > 0:
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    
    return {
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'f1_at_k': f1_at_k
    }
```

### 5.2 生成评估
```python
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

def calculate_generation_metrics(generated_answer, reference_answer):
    rouge = Rouge()
    
    # ROUGE分数
    rouge_scores = rouge.get_scores(generated_answer, reference_answer)[0]
    
    # BLEU分数
    reference_tokens = reference_answer.split()
    generated_tokens = generated_answer.split()
    bleu_score = sentence_bleu([reference_tokens], generated_tokens)
    
    return {
        'rouge_1': rouge_scores['rouge-1']['f'],
        'rouge_2': rouge_scores['rouge-2']['f'],
        'rouge_l': rouge_scores['rouge-l']['f'],
        'bleu': bleu_score
    }
```

## 6. 优化技术

### 6.1 自适应分块
```python
class AdaptiveChunker:
    def __init__(self, min_chunk_size=200, max_chunk_size=800):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_by_structure(self, text):
        # 按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= self.max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
```

### 6.2 缓存机制
```python
from functools import lru_cache
import hashlib

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
            # 删除最旧的条目
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[query_hash] = result
    
    def clear(self):
        self.cache.clear()
```

## 7. 部署和API

### 7.1 FastAPI部署
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
rag_system = None  # 全局RAG系统实例

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        if rag_system is None:
            raise HTTPException(status_code=500, detail="RAG系统未初始化")
        
        result = rag_system.query(request.question, request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 8. 监控和日志

### 8.1 性能监控
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

# ... existing code ...