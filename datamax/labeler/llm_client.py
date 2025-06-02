"""
大语言模型客户端

支持多种LLM服务的统一接口，包括API重试、错误处理、token计算等功能
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import openai
import tiktoken
from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMConfig(BaseModel):
    """LLM配置"""
    model_name: str = Field(default="gpt-3.5-turbo", description="模型名称")
    api_key: str = Field(description="API密钥")
    base_url: Optional[str] = Field(default=None, description="API基础URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(default=None, description="最大token数")
    timeout: int = Field(default=30, description="请求超时时间(秒)")
    max_retries: int = Field(default=3, description="最大重试次数")
    
    class Config:
        extra = "forbid"


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(description="角色: system/user/assistant")
    content: str = Field(description="消息内容")
    
    class Config:
        extra = "forbid"


class LLMResponse(BaseModel):
    """LLM响应"""
    content: str = Field(description="响应内容")
    usage: Dict[str, int] = Field(default_factory=dict, description="token使用统计")
    model: str = Field(description="使用的模型")
    finish_reason: str = Field(default="stop", description="完成原因")
    response_time: float = Field(description="响应时间(秒)")
    
    class Config:
        extra = "forbid"


class BaseLLMClient(ABC):
    """基础LLM客户端抽象类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logger
        
    @abstractmethod
    async def achat(self, messages: List[ChatMessage]) -> LLMResponse:
        """异步聊天"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[ChatMessage]) -> LLMResponse:
        """同步聊天"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """计算token数量"""
        pass
    
    def create_message(self, role: str, content: str) -> ChatMessage:
        """创建消息"""
        return ChatMessage(role=role, content=content)
    
    def create_system_message(self, content: str) -> ChatMessage:
        """创建系统消息"""
        return self.create_message("system", content)
    
    def create_user_message(self, content: str) -> ChatMessage:
        """创建用户消息"""
        return self.create_message("user", content)


class OpenAIClient(BaseLLMClient):
    """OpenAI客户端"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
        
        # 初始化异步客户端
        self.async_client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
        
        # 初始化tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.model_name)
        except KeyError:
            # 如果模型不在已知列表中，使用默认编码
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.logger.warning(f"模型 {config.model_name} 使用默认tokenizer")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def achat(self, messages: List[ChatMessage]) -> LLMResponse:
        """异步聊天"""
        start_time = time.time()
        
        try:
            # 转换消息格式
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # 发起请求
            response = await self.async_client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            response_time = time.time() - start_time
            
            # 解析响应
            choice = response.choices[0]
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            return LLMResponse(
                content=choice.message.content,
                usage=usage,
                model=response.model,
                finish_reason=choice.finish_reason,
                response_time=response_time
            )
            
        except Exception as e:
            self.logger.error(f"异步聊天请求失败: {str(e)}")
            raise
    
    def chat(self, messages: List[ChatMessage]) -> LLMResponse:
        """同步聊天"""
        start_time = time.time()
        
        try:
            # 转换消息格式
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # 发起请求
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            response_time = time.time() - start_time
            
            # 解析响应
            choice = response.choices[0]
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            return LLMResponse(
                content=choice.message.content,
                usage=usage,
                model=response.model,
                finish_reason=choice.finish_reason,
                response_time=response_time
            )
            
        except Exception as e:
            self.logger.error(f"同步聊天请求失败: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """计算token数量"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            self.logger.error(f"计算token失败: {str(e)}")
            # 粗略估算：1个token约等于4个字符（对于中文）
            return len(text) // 4


class TokenCounter:
    """Token计数器"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        初始化Token计数器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.logger = logger
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.logger.warning(f"模型 {model_name} 使用默认tokenizer")
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            int: token数量
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            self.logger.error(f"计算token失败: {str(e)}")
            # 粗略估算
            return len(text) // 4
    
    def count_messages_tokens(self, messages: List[ChatMessage]) -> int:
        """
        计算消息列表的token数量
        
        Args:
            messages: 消息列表
            
        Returns:
            int: 总token数量
        """
        total_tokens = 0
        
        for message in messages:
            # 每条消息的基础开销（role + content + 分隔符等）
            total_tokens += 4
            
            # 角色字段的token
            total_tokens += self.count_tokens(message.role)
            
            # 内容字段的token
            total_tokens += self.count_tokens(message.content)
        
        # 对话的基础开销
        total_tokens += 2
        
        return total_tokens
    
    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, 
                     model: str = None) -> Dict[str, float]:
        """
        估算API调用成本
        
        Args:
            prompt_tokens: 输入token数
            completion_tokens: 输出token数
            model: 模型名称
            
        Returns:
            Dict[str, float]: 成本信息
        """
        model = model or self.model_name
        
        # OpenAI定价（美元/1K tokens）
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }
        
        if model not in pricing:
            self.logger.warning(f"未知模型 {model}，使用gpt-3.5-turbo定价")
            model = "gpt-3.5-turbo"
        
        input_cost = (prompt_tokens / 1000) * pricing[model]["input"]
        output_cost = (completion_tokens / 1000) * pricing[model]["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "currency": "USD"
        }


class LLMClient:
    """LLM客户端工厂类"""
    
    @staticmethod
    def create_client(provider: str = "openai", 
                     config: Optional[LLMConfig] = None) -> BaseLLMClient:
        """
        创建LLM客户端
        
        Args:
            provider: 提供商 ('openai')
            config: LLM配置
            
        Returns:
            BaseLLMClient: LLM客户端实例
        """
        if config is None:
            raise ValueError("必须提供LLM配置")
        
        if provider.lower() == "openai":
            return OpenAIClient(config)
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")
    
    @staticmethod
    def create_openai_client(api_key: str, 
                           model_name: str = "gpt-3.5-turbo",
                           base_url: Optional[str] = None,
                           **kwargs) -> OpenAIClient:
        """
        快速创建OpenAI客户端
        
        Args:
            api_key: API密钥
            model_name: 模型名称
            base_url: API基础URL
            **kwargs: 其他配置参数
            
        Returns:
            OpenAIClient: OpenAI客户端实例
        """
        config = LLMConfig(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            **kwargs
        )
        return OpenAIClient(config) 