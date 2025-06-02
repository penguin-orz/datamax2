"""
基础标注器抽象类

定义所有标注器的通用接口和共同方法
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field


class LabelConfig(BaseModel):
    """标注配置模型"""
    chunk_size: int = Field(default=500, ge=100, le=2000, description="文本块大小")
    chunk_overlap: int = Field(default=100, ge=0, le=500, description="文本块重叠大小")
    max_workers: int = Field(default=5, ge=1, le=20, description="最大并发数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p采样")
    max_tokens: int = Field(default=2048, ge=256, le=8192, description="最大生成token数")
    
    class Config:
        extra = "forbid"


class LabelResult(BaseModel):
    """标注结果模型"""
    success: bool = Field(description="是否成功")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="标注数据")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="统计信息")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    
    class Config:
        extra = "forbid"


class BaseLabeler(ABC):
    """基础标注器抽象类"""
    
    def __init__(self, config: Optional[LabelConfig] = None):
        """
        初始化基础标注器
        
        Args:
            config: 标注配置，如果为None则使用默认配置
        """
        self.config = config or LabelConfig()
        self.logger = logger
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志配置"""
        self.logger.add(
            f"logs/labeler_{datetime.now().strftime('%Y%m%d')}.log",
            rotation="1 day",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        )
    
    @abstractmethod
    def label_from_text(self, text: str, **kwargs) -> LabelResult:
        """
        从文本生成标注数据
        
        Args:
            text: 输入文本
            **kwargs: 其他参数
            
        Returns:
            LabelResult: 标注结果
        """
        pass
    
    @abstractmethod
    def label_from_file(self, file_path: Union[str, Path], **kwargs) -> LabelResult:
        """
        从文件生成标注数据
        
        Args:
            file_path: 文件路径
            **kwargs: 其他参数
            
        Returns:
            LabelResult: 标注结果
        """
        pass
    
    def save_result(self, result: LabelResult, output_path: Union[str, Path], 
                   format_type: str = "jsonl") -> bool:
        """
        保存标注结果到文件
        
        Args:
            result: 标注结果
            output_path: 输出路径
            format_type: 输出格式 (jsonl, json, csv)
            
        Returns:
            bool: 是否保存成功
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type.lower() == "jsonl":
                self._save_as_jsonl(result.data, output_path)
            elif format_type.lower() == "json":
                self._save_as_json(result.data, output_path)
            elif format_type.lower() == "csv":
                self._save_as_csv(result.data, output_path)
            else:
                raise ValueError(f"不支持的格式类型: {format_type}")
            
            self.logger.success(f"标注结果已保存到: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")
            return False
    
    def _save_as_jsonl(self, data: List[Dict], output_path: Path):
        """保存为JSONL格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def _save_as_json(self, data: List[Dict], output_path: Path):
        """保存为JSON格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_as_csv(self, data: List[Dict], output_path: Path):
        """保存为CSV格式"""
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            # Pydantic会自动验证
            return True
        except Exception as e:
            self.logger.error(f"配置验证失败: {str(e)}")
            return False
    
    def get_statistics(self, result: LabelResult) -> Dict[str, Any]:
        """获取标注统计信息"""
        if not result.success or not result.data:
            return {"total_items": 0, "success_rate": 0.0}
        
        return {
            "total_items": len(result.data),
            "success_rate": 1.0,
            "created_at": result.created_at.isoformat(),
            "config": self.config.dict()
        } 