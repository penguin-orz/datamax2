"""
Token计数器

用于计算文本的token数量，支持多种编码方式
"""

import re
from typing import Dict, List, Optional, Union

import tiktoken
from loguru import logger
from pydantic import BaseModel, Field


class TokenCounterConfig(BaseModel):
    """Token计数器配置"""
    encoding_name: str = Field(default="cl100k_base", description="编码名称")
    model_name: Optional[str] = Field(default=None, description="模型名称")
    fallback_method: str = Field(default="estimate", description="回退方法")
    
    class Config:
        extra = "forbid"


class TokenCounter:
    """Token计数器类"""
    
    def __init__(self, config: Optional[TokenCounterConfig] = None):
        """
        初始化Token计数器
        
        Args:
            config: 配置对象
        """
        self.config = config or TokenCounterConfig()
        self.logger = logger
        self._encoding = None
        
        # 尝试初始化编码器
        self._init_encoding()
    
    def _init_encoding(self):
        """初始化编码器"""
        try:
            if self.config.model_name:
                # 根据模型名称获取编码器
                self._encoding = tiktoken.encoding_for_model(self.config.model_name)
                self.logger.info(f"使用模型 {self.config.model_name} 的编码器")
            else:
                # 使用指定的编码名称
                self._encoding = tiktoken.get_encoding(self.config.encoding_name)
                self.logger.info(f"使用编码器: {self.config.encoding_name}")
                
        except Exception as e:
            self.logger.warning(f"初始化tiktoken编码器失败: {str(e)}，将使用估算方法")
            self._encoding = None
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            int: token数量
        """
        if not text or not text.strip():
            return 0
        
        try:
            if self._encoding is not None:
                # 使用tiktoken进行精确计算
                tokens = self._encoding.encode(text)
                return len(tokens)
            else:
                # 使用回退方法估算
                return self._estimate_tokens(text)
                
        except Exception as e:
            self.logger.warning(f"Token计算失败: {str(e)}，使用估算方法")
            return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算token数量
        
        Args:
            text: 输入文本
            
        Returns:
            int: 估算的token数量
        """
        if self.config.fallback_method == "estimate":
            # 简单估算：英文约4个字符=1个token，中文约1.5个字符=1个token
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            english_chars = len(text) - chinese_chars
            
            # 中文字符 / 1.5 + 英文字符 / 4
            estimated_tokens = int(chinese_chars / 1.5 + english_chars / 4)
            return max(1, estimated_tokens)  # 至少返回1
            
        elif self.config.fallback_method == "char_count":
            # 基于字符数估算
            return max(1, len(text) // 3)
            
        elif self.config.fallback_method == "word_count":
            # 基于单词数估算
            words = text.split()
            return max(1, len(words))
            
        else:
            # 默认估算
            return max(1, len(text) // 4)
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        批量计算token数量
        
        Args:
            texts: 文本列表
            
        Returns:
            List[int]: token数量列表
        """
        return [self.count_tokens(text) for text in texts]
    
    def get_token_distribution(self, texts: List[str]) -> Dict[str, Union[int, float]]:
        """
        获取token分布统计
        
        Args:
            texts: 文本列表
            
        Returns:
            Dict: 统计信息
        """
        token_counts = self.count_tokens_batch(texts)
        
        if not token_counts:
            return {
                "total_texts": 0,
                "total_tokens": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0
            }
        
        return {
            "total_texts": len(texts),
            "total_tokens": sum(token_counts),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts)
        }
    
    def estimate_cost(self, text: str, 
                     input_price_per_1k: float = 0.01,
                     output_price_per_1k: float = 0.03) -> Dict[str, float]:
        """
        估算使用成本
        
        Args:
            text: 输入文本
            input_price_per_1k: 每1000个输入token的价格
            output_price_per_1k: 每1000个输出token的价格
            
        Returns:
            Dict: 成本估算信息
        """
        input_tokens = self.count_tokens(text)
        
        # 假设输出token数量为输入的30%
        estimated_output_tokens = int(input_tokens * 0.3)
        
        input_cost = (input_tokens / 1000) * input_price_per_1k
        output_cost = (estimated_output_tokens / 1000) * output_price_per_1k
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
    def chunk_by_tokens(self, text: str, max_tokens: int = 1000, 
                       overlap_tokens: int = 100) -> List[str]:
        """
        按token数量分割文本
        
        Args:
            text: 输入文本
            max_tokens: 每块最大token数
            overlap_tokens: 重叠token数
            
        Returns:
            List[str]: 分割后的文本块
        """
        if not text.strip():
            return []
        
        # 按句子分割
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # 如果单句就超过最大限制，需要进一步分割
            if sentence_tokens > max_tokens:
                # 保存当前块
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # 分割长句
                sub_chunks = self._split_long_sentence(sentence, max_tokens)
                chunks.extend(sub_chunks)
                
                current_chunk = ""
                current_tokens = 0
                continue
            
            # 检查是否可以添加到当前块
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk += sentence + " "
                current_tokens += sentence_tokens
            else:
                # 保存当前块并开始新块
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # 处理重叠
                if overlap_tokens > 0 and chunks:
                    overlap_text = self._get_overlap_text(current_chunk, overlap_tokens)
                    current_chunk = overlap_text + sentence + " "
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk = sentence + " "
                    current_tokens = sentence_tokens
        
        # 保存最后一块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        # 简单的句子分割，可以根据需要改进
        sentences = re.split(r'[。！？\.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str, max_tokens: int) -> List[str]:
        """分割长句"""
        # 按标点符号进一步分割
        parts = re.split(r'[，,；;：:]+', sentence)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            part_tokens = self.count_tokens(part)
            
            if self.count_tokens(current_chunk + part) <= max_tokens:
                current_chunk += part
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """获取重叠文本"""
        words = text.split()
        if not words:
            return ""
        
        # 从后往前取词，直到达到重叠token数量
        overlap_text = ""
        for word in reversed(words):
            test_text = word + " " + overlap_text
            if self.count_tokens(test_text) <= overlap_tokens:
                overlap_text = test_text
            else:
                break
        
        return overlap_text
    
    def get_encoding_info(self) -> Dict[str, Union[str, bool]]:
        """获取编码器信息"""
        return {
            "encoding_name": self.config.encoding_name,
            "model_name": self.config.model_name,
            "tiktoken_available": self._encoding is not None,
            "fallback_method": self.config.fallback_method
        } 