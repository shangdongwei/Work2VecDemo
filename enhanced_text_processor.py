import os
import re
import nltk
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedTextProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        初始化增强文本处理器
        
        Args:
            chunk_size: 文档分块大小（字符数）
            chunk_overlap: 分块重叠大小（字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 下载必要的NLTK数据
        self._download_nltk_data()
        
        logger.info(f"初始化文本处理器: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def _download_nltk_data(self):
        """下载必要的NLTK数据"""
        required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.download(data_name, quiet=True)
                except:
                    logger.warning(f"无法下载NLTK数据: {data_name}")
    
    def detect_file_encoding(self, file_path: str) -> str:
        """
        检测文件编码
        
        Args:
            file_path: 文件路径
            
        Returns:
            检测到的编码
        """
        import chardet
        
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read(10000)  # 读取前10KB用于检测
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                logger.info(f"检测到文件编码: {encoding} (置信度: {confidence:.2f})")
                
                # 如果置信度太低，使用默认编码
                if confidence < 0.7:
                    encoding = 'utf-8'
                    logger.warning(f"编码检测置信度较低，使用默认编码: {encoding}")
                
                return encoding
        except Exception as e:
            logger.error(f"编码检测失败: {e}，使用默认编码 utf-8")
            return 'utf-8'
    
    def clean_text(self, text: str) -> str:
        """
        清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符，但保留基本标点和换行
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\n\r]', '', text)
        
        # 移除过多的换行符
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 移除行首行尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        return text.strip()
    
    def extract_metadata_from_filename(self, file_path: str) -> Dict[str, Any]:
        """
        从文件名提取元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            元数据字典
        """
        path_obj = Path(file_path)
        
        metadata = {
            'filename': path_obj.name,
            'file_extension': path_obj.suffix,
            'file_size': path_obj.stat().st_size if path_obj.exists() else 0,
            'directory': str(path_obj.parent),
            'stem': path_obj.stem
        }
        
        # 尝试从文件名提取更多信息
        filename_lower = path_obj.stem.lower()
        
        # 检测可能的文档类型
        if any(keyword in filename_lower for keyword in ['readme', 'introduction', 'intro']):
            metadata['document_type'] = 'introduction'
        elif any(keyword in filename_lower for keyword in ['manual', 'guide', 'tutorial']):
            metadata['document_type'] = 'guide'
        elif any(keyword in filename_lower for keyword in ['api', 'reference', 'doc']):
            metadata['document_type'] = 'reference'
        elif any(keyword in filename_lower for keyword in ['faq', 'question', 'answer']):
            metadata['document_type'] = 'faq'
        else:
            metadata['document_type'] = 'general'
        
        return metadata
    
    def intelligent_chunking(self, text: str) -> List[str]:
        """
        智能文档分块，考虑语义边界
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        # 首先按段落分割
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 如果当前块加上新段落不超过限制，则添加
            if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # 如果当前块不为空，保存它
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果单个段落太长，需要进一步分割
                if len(paragraph) > self.chunk_size:
                    sub_chunks = self._split_long_paragraph(paragraph)
                    chunks.extend(sub_chunks[:-1])  # 除了最后一个
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = paragraph
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        # 添加重叠处理
        overlapped_chunks = self._add_overlap(chunks)
        
        return overlapped_chunks
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """
        分割过长的段落
        
        Args:
            paragraph: 长段落
            
        Returns:
            分割后的文本块列表
        """
        # 首先尝试按句子分割
        sentences = nltk.sent_tokenize(paragraph)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果单个句子太长，按字符强制分割
                if len(sentence) > self.chunk_size:
                    sub_chunks = self._force_split(sentence)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _force_split(self, text: str) -> List[str]:
        """
        强制按字符数分割文本
        
        Args:
            text: 输入文本
            
        Returns:
            分割后的文本块列表
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        为文本块添加重叠
        
        Args:
            chunks: 原始文本块列表
            
        Returns:
            添加重叠后的文本块列表
        """
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # 第一个块保持不变
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # 从前一个块的末尾提取重叠部分
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            # 将重叠部分添加到当前块的开头
            overlapped_chunk = overlap_text + " " + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def load_txt_file(self, file_path: str) -> Dict[str, Any]:
        """
        加载并处理单个txt文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含文档信息的字典
        """
        try:
            # 检测编码
            encoding = self.detect_file_encoding(file_path)
            
            # 读取文件
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            
            # 清理文本
            cleaned_content = self.clean_text(content)
            
            # 提取元数据
            metadata = self.extract_metadata_from_filename(file_path)
            
            # 智能分块
            chunks = self.intelligent_chunking(cleaned_content)
            
            # 构建文档信息
            document_info = {
                'file_path': file_path,
                'content': cleaned_content,
                'chunks': chunks,
                'metadata': metadata,
                'chunk_count': len(chunks),
                'total_length': len(cleaned_content)
            }
            
            logger.info(f"成功处理文件: {file_path}, 分割为 {len(chunks)} 个块")
            return document_info
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            return None
    
    def load_txt_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        批量加载并处理txt文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            文档信息列表
        """
        documents = []
        
        for file_path in file_paths:
            if not file_path.lower().endswith('.txt'):
                logger.warning(f"跳过非txt文件: {file_path}")
                continue
            
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                continue
            
            doc_info = self.load_txt_file(file_path)
            if doc_info:
                documents.append(doc_info)
        
        logger.info(f"成功处理 {len(documents)} 个txt文件")
        return documents
    
    def extract_document_structure(self, text: str) -> Dict[str, Any]:
        """
        提取文档结构信息
        
        Args:
            text: 文档文本
            
        Returns:
            文档结构信息
        """
        structure = {
            'headings': [],
            'paragraphs': 0,
            'sentences': 0,
            'words': 0,
            'has_lists': False,
            'has_code_blocks': False
        }
        
        # 检测标题（简单的启发式方法）
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                # 检测可能的标题
                if (len(line) < 100 and 
                    (line.isupper() or 
                     line.startswith('#') or 
                     re.match(r'^\d+\.', line) or
                     re.match(r'^[A-Z][^.!?]*$', line))):
                    structure['headings'].append(line)
        
        # 统计段落
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        structure['paragraphs'] = len(paragraphs)
        
        # 统计句子
        sentences = nltk.sent_tokenize(text)
        structure['sentences'] = len(sentences)
        
        # 统计单词
        words = text.split()
        structure['words'] = len(words)
        
        # 检测列表
        structure['has_lists'] = bool(re.search(r'^\s*[-*•]\s+', text, re.MULTILINE))
        
        # 检测代码块
        structure['has_code_blocks'] = bool(re.search(r'```|`[^`]+`', text))
        
        return structure

# 示例使用
if __name__ == "__main__":
    processor = EnhancedTextProcessor(chunk_size=800, chunk_overlap=100)
    print("增强文本处理器已创建完成！")