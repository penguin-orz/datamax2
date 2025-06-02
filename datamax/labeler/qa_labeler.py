"""
QA标注器

基于LLM的智能问答对生成器，支持多种文本类型和定制化需求
"""

import json
import re
from typing import Dict, List, Optional, Union
from datetime import datetime

from loguru import logger
from pydantic import BaseModel, Field

from .base_labeler import BaseLabeler, LabelConfig, LabelResult
from .llm_client import BaseLLMClient, ChatMessage
from .prompt_manager import PromptManager, SYSTEM_PROMPTS
from .text_splitter import TextSplitter, SplitterConfig


class QAPair(BaseModel):
    """问答对"""
    question: str = Field(description="问题")
    answer: str = Field(description="答案")
    type: str = Field(default="factual", description="问题类型")
    difficulty: str = Field(default="medium", description="难度级别")
    
    class Config:
        extra = "forbid"


class QAConfig(LabelConfig):
    """QA标注配置"""
    num_qa_per_chunk: int = Field(default=3, ge=1, le=10, description="每个文本块生成的QA对数量")
    question_types: List[str] = Field(
        default=["factual", "comprehension", "application"],
        description="问题类型列表"
    )
    difficulty_levels: List[str] = Field(
        default=["easy", "medium", "hard"],
        description="难度级别列表"
    )
    min_answer_length: int = Field(default=10, description="答案最小长度")
    max_answer_length: int = Field(default=500, description="答案最大长度")
    enable_filtering: bool = Field(default=True, description="是否启用质量过滤")
    
    class Config:
        extra = "forbid"


class QAResult(LabelResult):
    """QA标注结果"""
    qa_pairs: List[QAPair] = Field(default_factory=list, description="生成的QA对")
    source_text: str = Field(description="源文本")
    chunk_info: Dict = Field(default_factory=dict, description="文本块信息")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="时间戳")
    input_text: str = Field(default="", description="输入文本")
    config: Dict = Field(default_factory=dict, description="配置信息")
    
    class Config:
        extra = "forbid"


class QALabeler(BaseLabeler):
    """QA标注器"""
    
    def __init__(self, llm_client: BaseLLMClient, config: Optional[QAConfig] = None):
        """
        初始化QA标注器
        
        Args:
            llm_client: LLM客户端
            config: QA标注配置
        """
        if config is None:
            config = QAConfig()
        
        super().__init__(config)
        self.llm_client = llm_client
        self.prompt_manager = PromptManager()
        self.text_splitter = TextSplitter.create_splitter("recursive")
        
        # 设置系统提示词
        self.system_prompt = SYSTEM_PROMPTS["qa_generator"]
    
    def label_from_text(self, text: str) -> LabelResult:
        """
        从文本生成标注结果（基类抽象方法实现）
        
        Args:
            text: 输入文本
            
        Returns:
            LabelResult: 标注结果
        """
        qa_results = self.label_text(text)
        
        # 转换为基类要求的格式
        data = []
        for result in qa_results:
            for qa_pair in result.qa_pairs:
                data.append({
                    "question": qa_pair.question,
                    "answer": qa_pair.answer,
                    "type": qa_pair.type,
                    "difficulty": qa_pair.difficulty,
                    "source_text": result.source_text[:200] + "..." if len(result.source_text) > 200 else result.source_text,
                    "chunk_id": result.chunk_info.get("chunk_id"),
                    "timestamp": result.timestamp
                })
        
        return LabelResult(
            success=len(data) > 0,
            data=data,
            error_message=None if len(data) > 0 else "未生成QA对",
            statistics={"total_qa_pairs": len(data)}
        )
    
    def label_from_file(self, file_path: str) -> LabelResult:
        """
        从文件生成标注结果（基类抽象方法实现）
        
        Args:
            file_path: 文件路径
            
        Returns:
            LabelResult: 标注结果
        """
        qa_results = self.generate_qa_from_file(file_path)
        
        # 转换为基类要求的格式
        data = []
        for result in qa_results:
            for qa_pair in result.qa_pairs:
                data.append({
                    "question": qa_pair.question,
                    "answer": qa_pair.answer,
                    "type": qa_pair.type,
                    "difficulty": qa_pair.difficulty,
                    "source_file": file_path,
                    "chunk_id": result.chunk_info.get("chunk_id"),
                    "timestamp": result.timestamp
                })
        
        return LabelResult(
            success=len(data) > 0,
            data=data,
            error_message=None if len(data) > 0 else "未生成QA对",
            statistics={"total_qa_pairs": len(data), "source_file": file_path}
        )
    
    def label_text(self, text: str) -> List[QAResult]:
        """
        对文本进行QA标注
        
        Args:
            text: 输入文本
            
        Returns:
            List[QAResult]: QA标注结果列表
        """
        try:
            # 分割文本
            chunks = self._split_text(text)
            
            results = []
            for i, chunk in enumerate(chunks):
                try:
                    # 生成QA对
                    qa_pairs = self._generate_qa_pairs(chunk.content)
                    
                    # 过滤QA对
                    if self.config.enable_filtering:
                        qa_pairs = self._filter_qa_pairs(qa_pairs)
                    
                    # 创建结果
                    result = QAResult(
                        success=len(qa_pairs) > 0,
                        data=[],  # 将在转换时填充
                        input_text=chunk.content,
                        qa_pairs=qa_pairs,
                        source_text=text,
                        chunk_info={
                            "chunk_id": chunk.chunk_id,
                            "chunk_size": len(chunk.content),
                            "metadata": chunk.metadata
                        },
                        timestamp=datetime.now().isoformat(),
                        config=self.config.dict()
                    )
                    
                    results.append(result)
                    
                    self.logger.info(f"完成第 {i+1}/{len(chunks)} 个文本块的QA生成，生成 {len(qa_pairs)} 个QA对")
                    
                except Exception as e:
                    self.logger.error(f"文本块 {i} QA生成失败: {str(e)}")
                    continue
            
            self.logger.info(f"QA标注完成，共处理 {len(chunks)} 个文本块，生成 {len(results)} 个结果")
            return results
            
        except Exception as e:
            self.logger.error(f"QA标注失败: {str(e)}")
            return []
    
    def _split_text(self, text: str):
        """分割文本"""
        # 根据配置创建分割器
        splitter_config = SplitterConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        splitter = TextSplitter.create_splitter("recursive", splitter_config)
        return splitter.split_text(text)
    
    def _generate_qa_pairs(self, text: str) -> List[QAPair]:
        """生成QA对"""
        try:
            # 构建提示词
            prompt = self.prompt_manager.render_template(
                "qa_generation",
                text=text,
                num_qa=self.config.num_qa_per_chunk
            )
            
            # 构建消息
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=prompt)
            ]
            
            # 调用LLM
            response = self.llm_client.chat(messages)
            
            # 解析响应
            qa_pairs = self._parse_qa_response(response.content)
            
            return qa_pairs
            
        except Exception as e:
            self.logger.error(f"QA对生成失败: {str(e)}")
            return []
    
    def _parse_qa_response(self, response: str) -> List[QAPair]:
        """解析QA响应"""
        try:
            # 提取JSON部分
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 如果没有代码块，尝试直接解析
                json_str = response
            
            # 解析JSON
            data = json.loads(json_str)
            
            qa_pairs = []
            for qa_data in data.get("qa_pairs", []):
                qa_pair = QAPair(
                    question=qa_data.get("question", ""),
                    answer=qa_data.get("answer", ""),
                    type=qa_data.get("type", "factual")
                )
                qa_pairs.append(qa_pair)
            
            return qa_pairs
            
        except Exception as e:
            self.logger.error(f"解析QA响应失败: {str(e)}")
            return []
    
    def _filter_qa_pairs(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """过滤QA对"""
        filtered_pairs = []
        
        for qa_pair in qa_pairs:
            # 检查问题和答案是否为空
            if not qa_pair.question.strip() or not qa_pair.answer.strip():
                continue
            
            # 检查答案长度
            answer_length = len(qa_pair.answer)
            if answer_length < self.config.min_answer_length or answer_length > self.config.max_answer_length:
                continue
            
            # 检查问题质量
            if not self._is_good_question(qa_pair.question):
                continue
            
            filtered_pairs.append(qa_pair)
        
        return filtered_pairs
    
    def _is_good_question(self, question: str) -> bool:
        """检查问题质量"""
        # 基本长度检查
        if len(question) < 5:
            return False
        
        # 检查是否以问号结尾
        if not question.strip().endswith(('?', '？')):
            return False
        
        # 检查是否包含疑问词
        question_words = ['什么', '为什么', '如何', '怎样', '哪里', '谁', '什么时候', '多少']
        if not any(word in question for word in question_words):
            return False
        
        return True
    
    def batch_label_texts(self, texts: List[str]) -> List[List[QAResult]]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[QAResult]]: 批量QA标注结果
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                text_results = self.label_text(text)
                results.append(text_results)
                
                self.logger.info(f"完成第 {i+1}/{len(texts)} 个文本的处理")
                
            except Exception as e:
                self.logger.error(f"文本 {i} 处理失败: {str(e)}")
                results.append([])
        
        return results
    
    def generate_qa_from_file(self, file_path: str) -> List[QAResult]:
        """
        从文件生成QA对
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[QAResult]: QA标注结果列表
        """
        try:
            # 使用文本分割器处理文件
            chunks = self.text_splitter.split_file(file_path)
            
            results = []
            for chunk in chunks:
                chunk_results = self.label_text(chunk.content)
                
                # 更新源文件信息
                for result in chunk_results:
                    result.chunk_info.update({
                        "source_file": file_path,
                        "original_chunk_id": chunk.chunk_id
                    })
                
                results.extend(chunk_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"从文件生成QA失败: {str(e)}")
            return []
    
    def export_qa_dataset(self, results: List[QAResult], 
                         output_path: str, format: str = "jsonl"):
        """
        导出QA数据集
        
        Args:
            results: QA结果列表
            output_path: 输出路径
            format: 输出格式 ('jsonl', 'json', 'csv')
        """
        try:
            # 收集所有QA对
            all_qa_pairs = []
            for result in results:
                for qa_pair in result.qa_pairs:
                    qa_data = {
                        "question": qa_pair.question,
                        "answer": qa_pair.answer,
                        "type": qa_pair.type,
                        "difficulty": qa_pair.difficulty,
                        "source_text": result.source_text[:200] + "..." if len(result.source_text) > 200 else result.source_text,
                        "chunk_id": result.chunk_info.get("chunk_id"),
                        "timestamp": result.timestamp
                    }
                    all_qa_pairs.append(qa_data)
            
            # 根据格式保存
            if format.lower() == "jsonl":
                with open(output_path, 'w', encoding='utf-8') as f:
                    for qa_data in all_qa_pairs:
                        f.write(json.dumps(qa_data, ensure_ascii=False) + '\n')
            
            elif format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
            
            elif format.lower() == "csv":
                import pandas as pd
                df = pd.DataFrame(all_qa_pairs)
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"QA数据集已导出到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"导出QA数据集失败: {str(e)}")
            raise 