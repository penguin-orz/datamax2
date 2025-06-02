"""
DataMax语料标注模块

提供智能语料标注、QA对生成、文本分块等功能，支持多种数据格式和导出方式。

主要功能：
- 基于LLM的智能QA对生成
- 多种文本分割策略
- 灵活的提示词管理
- 精确的token计数
- 多格式数据导出

使用示例：
    from datamax.labeler import QALabeler, TextSplitter
    
    # 创建QA标注器
    qa_labeler = QALabeler(llm_client)
    
    # 生成QA对
    results = qa_labeler.label_text("这里是要标注的文本")
"""

from .base_labeler import BaseLabeler, LabelConfig, LabelResult
from .llm_client import BaseLLMClient, LLMClient
from .prompt_manager import PromptManager
from .qa_labeler import QALabeler, QAConfig, QAPair, QAResult
from .text_splitter import TextSplitter, SplitterConfig, TextChunk
from .token_counter import TokenCounter, TokenCounterConfig

__all__ = [
    # 基础类
    "BaseLabeler",
    "LabelConfig", 
    "LabelResult",
    
    # LLM客户端
    "BaseLLMClient",
    "LLMClient",
    
    # QA标注
    "QALabeler",
    "QAConfig",
    "QAPair",
    "QAResult",
    
    # 文本分割
    "TextSplitter",
    "SplitterConfig", 
    "TextChunk",
    
    # 提示词管理
    "PromptManager",
    
    # Token计数
    "TokenCounter",
    "TokenCounterConfig",
]

__version__ = "1.0.0" 