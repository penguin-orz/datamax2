"""
文本分割器

支持多种文本分割策略，包括按长度、按语义、按段落等
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from loguru import logger
from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """文本块模型"""
    content: str = Field(description="文本内容")
    metadata: dict = Field(default_factory=dict, description="元数据")
    chunk_id: Optional[int] = Field(default=None, description="块ID")
    
    class Config:
        extra = "forbid"


class SplitterConfig(BaseModel):
    """分割器配置"""
    chunk_size: int = Field(default=500, ge=100, le=5000, description="块大小")
    chunk_overlap: int = Field(default=100, ge=0, le=1000, description="重叠大小")
    separators: Optional[List[str]] = Field(default=None, description="分隔符列表")
    keep_separator: bool = Field(default=True, description="是否保留分隔符")
    
    class Config:
        extra = "forbid"


class BaseSplitter(ABC):
    """基础分割器抽象类"""
    
    def __init__(self, config: SplitterConfig):
        self.config = config
        self.logger = logger
    
    @abstractmethod
    def split_text(self, text: str) -> List[TextChunk]:
        """分割文本"""
        pass
    
    def split_file(self, file_path: Union[str, Path]) -> List[TextChunk]:
        """分割文件"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 根据文件类型选择加载方式
            if file_path.suffix.lower() == '.md':
                return self._split_markdown_file(file_path)
            else:
                # 默认按文本文件处理
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return self.split_text(content)
                
        except Exception as e:
            self.logger.error(f"分割文件失败: {str(e)}")
            return []
    
    def _split_markdown_file(self, file_path: Path) -> List[TextChunk]:
        """分割Markdown文件"""
        try:
            # 尝试使用UnstructuredMarkdownLoader
            try:
                loader = UnstructuredMarkdownLoader(str(file_path))
                documents = loader.load()
                
                chunks = []
                for doc in documents:
                    text_chunks = self.split_text(doc.page_content)
                    for chunk in text_chunks:
                        chunk.metadata.update({
                            'source_file': str(file_path),
                            'file_type': 'markdown'
                        })
                    chunks.extend(text_chunks)
                
                return chunks
            
            except Exception as loader_error:
                # 如果UnstructuredMarkdownLoader失败，回退到普通文本处理
                self.logger.warning(f"UnstructuredMarkdownLoader失败，回退到普通文本处理: {str(loader_error)}")
                
                # 直接读取文件内容并处理
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks = self.split_text(content)
                for chunk in chunks:
                    chunk.metadata.update({
                        'source_file': str(file_path),
                        'file_type': 'markdown',
                        'fallback_method': True
                    })
                
                return chunks
            
        except Exception as e:
            self.logger.error(f"分割Markdown文件失败: {str(e)}")
            return []


class RecursiveSplitter(BaseSplitter):
    """递归字符分割器"""
    
    def __init__(self, config: SplitterConfig):
        super().__init__(config)
        
        # 设置默认分隔符
        separators = config.separators or [
            "\n\n",  # 段落分隔
            "\n",    # 行分隔
            " ",     # 空格
            ""       # 字符级别
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=separators,
            keep_separator=config.keep_separator,
            length_function=len,
            is_separator_regex=False,
        )
    
    def split_text(self, text: str) -> List[TextChunk]:
        """递归分割文本"""
        try:
            docs = self.splitter.create_documents([text])
            chunks = []
            
            for i, doc in enumerate(docs):
                chunk = TextChunk(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    chunk_id=i
                )
                chunks.append(chunk)
            
            self.logger.info(f"文本分割完成，共生成 {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            self.logger.error(f"递归分割失败: {str(e)}")
            return []


class SemanticSplitter(BaseSplitter):
    """语义分割器 - 基于段落和语义边界"""
    
    def split_text(self, text: str) -> List[TextChunk]:
        """基于语义边界分割文本"""
        try:
            # 预处理：按段落分割
            paragraphs = self._split_by_paragraphs(text)
            
            chunks = []
            current_chunk = ""
            current_size = 0
            chunk_id = 0
            
            for paragraph in paragraphs:
                paragraph_size = len(paragraph)
                
                # 如果当前段落本身就超过块大小，需要进一步分割
                if paragraph_size > self.config.chunk_size:
                    # 保存当前块（如果有内容）
                    if current_chunk.strip():
                        chunks.append(TextChunk(
                            content=current_chunk.strip(),
                            metadata={"split_type": "semantic"},
                            chunk_id=chunk_id
                        ))
                        chunk_id += 1
                    
                    # 分割大段落
                    sub_chunks = self._split_large_paragraph(paragraph)
                    for sub_chunk in sub_chunks:
                        chunks.append(TextChunk(
                            content=sub_chunk,
                            metadata={"split_type": "semantic", "large_paragraph": True},
                            chunk_id=chunk_id
                        ))
                        chunk_id += 1
                    
                    current_chunk = ""
                    current_size = 0
                
                # 检查是否可以添加到当前块
                elif current_size + paragraph_size <= self.config.chunk_size:
                    current_chunk += paragraph + "\n\n"
                    current_size += paragraph_size + 2  # +2 for \n\n
                
                else:
                    # 保存当前块并开始新块
                    if current_chunk.strip():
                        chunks.append(TextChunk(
                            content=current_chunk.strip(),
                            metadata={"split_type": "semantic"},
                            chunk_id=chunk_id
                        ))
                        chunk_id += 1
                    
                    current_chunk = paragraph + "\n\n"
                    current_size = paragraph_size + 2
            
            # 保存最后一块
            if current_chunk.strip():
                chunks.append(TextChunk(
                    content=current_chunk.strip(),
                    metadata={"split_type": "semantic"},
                    chunk_id=chunk_id
                ))
            
            self.logger.info(f"语义分割完成，共生成 {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            self.logger.error(f"语义分割失败: {str(e)}")
            return []
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        # 按双换行符分割段落
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """分割大段落"""
        # 使用递归分割器处理大段落
        temp_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=min(self.config.chunk_overlap, self.config.chunk_size // 4),
            length_function=len,
        )
        
        docs = temp_splitter.create_documents([paragraph])
        return [doc.page_content for doc in docs]


class TextSplitter:
    """文本分割器工厂类"""
    
    @staticmethod
    def create_splitter(splitter_type: str = "recursive", 
                       config: Optional[SplitterConfig] = None) -> BaseSplitter:
        """
        创建分割器
        
        Args:
            splitter_type: 分割器类型 ('recursive', 'semantic')
            config: 分割器配置
            
        Returns:
            BaseSplitter: 分割器实例
        """
        if config is None:
            config = SplitterConfig()
        
        if splitter_type.lower() == "recursive":
            return RecursiveSplitter(config)
        elif splitter_type.lower() == "semantic":
            return SemanticSplitter(config)
        else:
            raise ValueError(f"不支持的分割器类型: {splitter_type}")
    
    @staticmethod
    def quick_split(text: str, chunk_size: int = 500, 
                   chunk_overlap: int = 100) -> List[str]:
        """
        快速分割文本（返回字符串列表）
        
        Args:
            text: 输入文本
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            
        Returns:
            List[str]: 分割后的文本块列表
        """
        config = SplitterConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splitter = RecursiveSplitter(config)
        chunks = splitter.split_text(text)
        return [chunk.content for chunk in chunks] 