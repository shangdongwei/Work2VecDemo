import dashscope
from dashscope import Generation
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class QwenLLM:
    def __init__(self, api_key: str = None, model_name: str = "qwen-turbo"):
        """
        初始化Qwen大语言模型
        
        Args:
            api_key: 阿里云DashScope API密钥
            model_name: 模型名称 (qwen-turbo, qwen-plus, qwen-max等)
        """
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("请设置DASHSCOPE_API_KEY环境变量或传入api_key参数")
        
        # 设置API密钥
        dashscope.api_key = self.api_key
        
        logger.info(f"初始化Qwen模型: {self.model_name}")
    
    def generate_response(self, 
                         prompt: str, 
                         max_tokens: int = 2000,
                         temperature: float = 0.7,
                         top_p: float = 0.8,
                         system_message: str = None) -> str:
        """
        生成回复
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数，控制随机性
            top_p: top-p采样参数
            system_message: 系统消息
            
        Returns:
            生成的回复文本
        """
        try:
            messages = []
            
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            response = Generation.call(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                result_format='message'
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                logger.error(f"API调用失败: {response.status_code}, {response.message}")
                return f"抱歉，生成回复时出现错误: {response.message}"
                
        except Exception as e:
            logger.error(f"生成回复时出错: {e}")
            return f"抱歉，生成回复时出现错误: {str(e)}"
    
    def generate_rag_response(self, 
                             query: str, 
                             context_documents: List[str],
                             max_tokens: int = 2000,
                             temperature: float = 0.7) -> str:
        """
        基于检索到的文档生成RAG回复
        
        Args:
            query: 用户查询
            context_documents: 检索到的相关文档列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            生成的回复
        """
        # 构建上下文
        context = "\n\n".join([f"文档{i+1}: {doc}" for i, doc in enumerate(context_documents)])
        
        # 构建RAG提示模板
        rag_prompt = f"""基于以下提供的文档内容，请回答用户的问题。如果文档中没有相关信息，请明确说明。

相关文档:
{context}

用户问题: {query}

请基于上述文档内容提供准确、详细的回答："""

        system_message = """你是一个专业的AI助手，擅长基于提供的文档内容回答问题。请确保：
1. 回答基于提供的文档内容
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、详细且有帮助
4. 可以适当引用文档中的具体内容"""

        return self.generate_response(
            prompt=rag_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_message=system_message
        )
    
    def summarize_document(self, document: str, max_tokens: int = 500) -> str:
        """
        文档摘要生成
        
        Args:
            document: 要摘要的文档
            max_tokens: 最大生成token数
            
        Returns:
            文档摘要
        """
        prompt = f"""请为以下文档生成一个简洁的摘要，突出主要内容和关键信息：

文档内容:
{document}

摘要:"""

        return self.generate_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            system_message="你是一个专业的文档摘要助手，能够提取文档的核心内容并生成简洁准确的摘要。"
        )
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            num_keywords: 要提取的关键词数量
            
        Returns:
            关键词列表
        """
        prompt = f"""请从以下文本中提取{num_keywords}个最重要的关键词，用逗号分隔：

文本内容:
{text}

关键词:"""

        response = self.generate_response(
            prompt=prompt,
            max_tokens=200,
            temperature=0.3,
            system_message="你是一个专业的关键词提取助手，能够识别文本中最重要的概念和术语。"
        )
        
        # 解析关键词
        keywords = [kw.strip() for kw in response.split(',')]
        return keywords[:num_keywords]
    
    def check_relevance(self, query: str, document: str) -> float:
        """
        检查文档与查询的相关性
        
        Args:
            query: 查询文本
            document: 文档文本
            
        Returns:
            相关性分数 (0-1之间)
        """
        prompt = f"""请评估以下文档与查询的相关性，给出0-10分的评分（10分表示高度相关，0分表示完全不相关）：

查询: {query}

文档: {document}

请只返回数字评分："""

        try:
            response = self.generate_response(
                prompt=prompt,
                max_tokens=50,
                temperature=0.1,
                system_message="你是一个专业的文档相关性评估助手，能够准确评估文档与查询的相关程度。"
            )
            
            # 提取数字评分
            import re
            score_match = re.search(r'\d+', response)
            if score_match:
                score = int(score_match.group())
                return min(score / 10.0, 1.0)  # 转换为0-1范围
            else:
                return 0.5  # 默认中等相关性
                
        except Exception as e:
            logger.error(f"评估相关性时出错: {e}")
            return 0.5

# 示例使用
if __name__ == "__main__":
    # 注意：需要设置DASHSCOPE_API_KEY环境变量
    try:
        # qwen_llm = QwenLLM(model_name="qwen-turbo")
        # response = qwen_llm.generate_response("你好，请介绍一下人工智能的发展历程。")
        # print("Qwen回复:", response)
        print("Qwen LLM类已创建完成！")
        print("使用前请确保设置了DASHSCOPE_API_KEY环境变量")
    except Exception as e:
        print(f"初始化Qwen模型时出错: {e}")
        print("请确保设置了正确的DASHSCOPE_API_KEY环境变量")