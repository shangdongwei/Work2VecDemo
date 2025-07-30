import os
import logging
from rag_system import RAGSystem

# 设置日志级别
logging.basicConfig(level=logging.INFO)

def create_sample_documents():
    """创建示例英文文档"""
    
    # 示例文档1：机器学习介绍
    doc1_content = """
Machine Learning: An Introduction

Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.

Types of Machine Learning:

1. Supervised Learning: This type of machine learning uses labeled datasets to train algorithms that classify data or predict outcomes accurately. As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately.

2. Unsupervised Learning: This type of machine learning uses machine learning algorithms to analyze and cluster unlabeled datasets. These algorithms discover hidden patterns or data groupings without the need for human intervention.

3. Reinforcement Learning: This is a type of machine learning where an agent learns to behave in an environment by performing actions and seeing the results. The agent receives rewards or penalties for the actions it performs.

Applications of machine learning include email filtering, detection of network intruders, and computer vision. Machine learning is closely related to computational statistics, which focuses on making predictions using computers.
"""

    # 示例文档2：深度学习
    doc2_content = """
Deep Learning: Neural Networks and Beyond

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.

Deep learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs.

Neural Networks:
A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. Neural networks can adapt to changing input; so the network generates the best possible result without needing to redesign the output criteria.

Convolutional Neural Networks (CNNs):
CNNs are a class of deep neural networks, most commonly applied to analyzing visual imagery. They are also known as shift invariant or space invariant artificial neural networks, based on their shared-weights architecture and translation invariance characteristics.

Recurrent Neural Networks (RNNs):
RNNs are a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Unlike feedforward neural networks, RNNs can use their internal state to process sequences of inputs.

The field of deep learning has seen tremendous growth in recent years, with applications ranging from autonomous vehicles to medical diagnosis, and from natural language understanding to game playing.
"""

    # 示例文档3：自然语言处理
    doc3_content = """
Natural Language Processing: Understanding Human Language

Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.

The goal of NLP is to enable computers to understand, interpret and manipulate human language. This involves several challenges including speech recognition, natural language understanding, and natural language generation.

Key NLP Tasks:

1. Tokenization: The process of breaking down text into individual words or tokens.

2. Part-of-Speech Tagging: Identifying the grammatical parts of speech for each word in a sentence.

3. Named Entity Recognition (NER): Identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.

4. Sentiment Analysis: Determining the emotional tone behind a series of words, used to gain an understanding of the attitudes, opinions and emotions expressed within an online mention.

5. Machine Translation: Automatically translating text from one language to another.

6. Question Answering: Building systems that automatically answer questions posed by humans in a natural language.

7. Text Summarization: Creating a short, accurate, and fluent summary of a longer text document.

Modern NLP techniques heavily rely on machine learning and deep learning approaches. Transformer models like BERT, GPT, and T5 have revolutionized the field by achieving state-of-the-art results on many NLP benchmarks.

Applications of NLP include chatbots, language translation services, sentiment analysis for social media monitoring, and information extraction from documents.
"""

    # 创建文档文件
    os.makedirs("sample_docs", exist_ok=True)
    
    with open("sample_docs/machine_learning.txt", "w", encoding="utf-8") as f:
        f.write(doc1_content)
    
    with open("sample_docs/deep_learning.txt", "w", encoding="utf-8") as f:
        f.write(doc2_content)
    
    with open("sample_docs/nlp.txt", "w", encoding="utf-8") as f:
        f.write(doc3_content)
    
    print("示例文档已创建在 sample_docs/ 目录中")
    return ["sample_docs/machine_learning.txt", "sample_docs/deep_learning.txt", "sample_docs/nlp.txt"]

def demo_rag_system():
    """演示RAG系统的使用"""
    
    print("=== RAG系统演示 ===\n")
    
    # 检查API密钥
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量")
        print("请设置您的阿里云DashScope API密钥以使用Qwen模型")
        print("示例: set DASHSCOPE_API_KEY=your_api_key_here")
        print("\n继续演示其他功能...\n")
    
    try:
        # 1. 创建示例文档
        print("1. 创建示例文档...")
        file_paths = create_sample_documents()
        
        # 2. 初始化RAG系统
        print("2. 初始化RAG系统...")
        rag = RAGSystem(
            vector_size=300,
            index_type="flat",
            qwen_model="qwen-turbo",
            api_key=api_key
        )
        
        # 3. 构建知识库
        print("3. 构建知识库...")
        rag.build_knowledge_base(file_paths, save_path="demo_rag")
        
        # 4. 显示系统统计信息
        print("4. 系统统计信息:")
        stats = rag.get_system_stats()
        print(f"   - 文档数量: {stats['vector_database']['active_documents']}")
        print(f"   - 词汇表大小: {stats['embedding_model']['vocabulary_size']}")
        print(f"   - 向量维度: {stats['embedding_model']['vector_size']}")
        
        # 5. 测试查询（如果有API密钥）
        if api_key:
            print("\n5. 测试查询:")
            
            test_queries = [
                "What is machine learning?",
                "Explain the difference between supervised and unsupervised learning",
                "What are neural networks?",
                "How does natural language processing work?",
                "What is the difference between CNN and RNN?"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n查询 {i}: {query}")
                print("-" * 50)
                
                result = rag.query(query, top_k=3)
                
                print(f"答案: {result['answer'][:200]}...")
                print(f"置信度: {result['confidence']:.3f}")
                print(f"检索到 {len(result['retrieved_documents'])} 个相关文档")
                
                # 显示检索到的文档片段
                for j, doc in enumerate(result['retrieved_documents'][:2]):
                    print(f"  文档{j+1} (相似度: {doc['similarity_score']:.3f}): {doc['content'][:100]}...")
        
        else:
            print("\n5. 跳过查询测试（需要API密钥）")
            print("   可以测试向量检索功能:")
            
            # 测试向量检索（不需要API密钥）
            query_embedding = rag.embedding_model.get_sentence_embedding("What is machine learning?")
            search_results = rag.vector_db.search(query_embedding, k=3)
            
            print(f"   检索到 {len(search_results)} 个相关文档:")
            for i, (doc, score, metadata) in enumerate(search_results):
                print(f"   文档{i+1} (相似度: {score:.3f}): {doc[:100]}...")
        
        # 6. 保存和加载测试
        print("\n6. 测试保存和加载功能...")
        rag.save_system("demo_rag")
        
        # 创建新的RAG实例并加载
        new_rag = RAGSystem(api_key=api_key)
        new_rag.load_system("demo_rag")
        
        print("   系统保存和加载成功！")
        
        print("\n=== 演示完成 ===")
        print("\n使用说明:")
        print("1. 将您的英文文档放在指定目录")
        print("2. 调用 rag.build_knowledge_base(['file1.txt', 'file2.txt']) 构建知识库")
        print("3. 调用 rag.query('your question') 进行查询")
        print("4. 使用 rag.save_system('path') 保存系统")
        print("5. 使用 rag.load_system('path') 加载系统")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_rag_system()