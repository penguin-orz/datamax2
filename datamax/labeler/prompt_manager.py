"""
提示词管理器

支持模板化提示词管理，包括QA生成、文本分类、实体识别等任务
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from jinja2 import Environment, Template
from loguru import logger
from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """提示词模板"""
    name: str = Field(description="模板名称")
    description: str = Field(description="模板描述")
    template: str = Field(description="模板内容")
    variables: List[str] = Field(default_factory=list, description="模板变量")
    category: str = Field(default="general", description="模板分类")
    language: str = Field(default="zh", description="语言")
    
    class Config:
        extra = "forbid"


class PromptManager:
    """提示词管理器"""
    
    def __init__(self, templates_dir: Optional[Union[str, Path]] = None):
        """
        初始化提示词管理器
        
        Args:
            templates_dir: 模板文件目录
        """
        self.templates: Dict[str, PromptTemplate] = {}
        self.env = Environment()
        self.logger = logger
        
        # 加载默认模板
        self._load_default_templates()
        
        # 如果指定了模板目录，加载文件中的模板
        if templates_dir:
            self.load_templates_from_dir(templates_dir)
    
    def _load_default_templates(self):
        """加载默认模板"""
        
        # QA生成模板
        qa_template = PromptTemplate(
            name="qa_generation",
            description="基于文本内容生成问答对",
            template="""你是一个专业的教育内容创作者。请基于以下文本内容，生成高质量的问答对。

## 文本内容：
{{text}}

## 要求：
1. 生成 {{num_qa}} 个问答对
2. 问题应该具有教育价值，能够帮助读者理解和记忆关键信息
3. 答案应该准确、完整，直接来源于原文
4. 问题类型应该多样化：事实性问题、理解性问题、应用性问题等
5. 避免生成过于简单或过于复杂的问题

## 输出格式：
请以JSON格式输出，结构如下：
```json
{
    "qa_pairs": [
        {
            "question": "问题内容",
            "answer": "答案内容",
            "type": "问题类型(factual/comprehension/application)"
        }
    ]
}
```

请开始生成：""",
            variables=["text", "num_qa"],
            category="qa",
            language="zh"
        )
        
        # 文本分类模板
        classification_template = PromptTemplate(
            name="text_classification",
            description="文本分类任务",
            template="""你是一个专业的文本分类专家。请对以下文本进行分类。

## 文本内容：
{{text}}

## 分类标签：
{{labels}}

## 要求：
1. 仔细阅读文本内容
2. 从给定的标签中选择最合适的分类
3. 提供分类的理由

## 输出格式：
```json
{
    "label": "分类标签",
    "confidence": 0.95,
    "reason": "分类理由"
}
```

请开始分类：""",
            variables=["text", "labels"],
            category="classification",
            language="zh"
        )
        
        # 实体识别模板
        ner_template = PromptTemplate(
            name="named_entity_recognition",
            description="命名实体识别",
            template="""你是一个专业的命名实体识别专家。请从以下文本中识别出所有的命名实体。

## 文本内容：
{{text}}

## 实体类型：
{{entity_types}}

## 要求：
1. 识别文本中的所有命名实体
2. 为每个实体标注类型
3. 提供实体在文本中的位置

## 输出格式：
```json
{
    "entities": [
        {
            "text": "实体文本",
            "label": "实体类型",
            "start": 开始位置,
            "end": 结束位置
        }
    ]
}
```

请开始识别：""",
            variables=["text", "entity_types"],
            category="ner",
            language="zh"
        )
        
        # 文本摘要模板
        summary_template = PromptTemplate(
            name="text_summarization",
            description="文本摘要生成",
            template="""你是一个专业的文本摘要专家。请为以下文本生成高质量的摘要。

## 文本内容：
{{text}}

## 摘要要求：
- 长度：{{max_length}} 个字符以内
- 风格：{{style}}
- 保留关键信息和核心观点
- 语言简洁明了

## 输出格式：
```json
{
    "summary": "摘要内容",
    "key_points": ["关键点1", "关键点2", "关键点3"]
}
```

请开始生成摘要：""",
            variables=["text", "max_length", "style"],
            category="summarization",
            language="zh"
        )
        
        # 注册默认模板
        for template in [qa_template, classification_template, ner_template, summary_template]:
            self.register_template(template)
    
    def register_template(self, template: PromptTemplate):
        """注册模板"""
        self.templates[template.name] = template
        self.logger.info(f"注册模板: {template.name}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """获取模板"""
        return self.templates.get(name)
    
    def render_template(self, name: str, **kwargs) -> str:
        """
        渲染模板
        
        Args:
            name: 模板名称
            **kwargs: 模板变量
            
        Returns:
            str: 渲染后的文本
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"模板不存在: {name}")
        
        try:
            jinja_template = Template(template.template)
            rendered = jinja_template.render(**kwargs)
            return rendered
        except Exception as e:
            self.logger.error(f"模板渲染失败: {str(e)}")
            raise
    
    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """
        列出模板
        
        Args:
            category: 筛选分类
            
        Returns:
            List[str]: 模板名称列表
        """
        if category:
            return [
                name for name, template in self.templates.items()
                if template.category == category
            ]
        else:
            return list(self.templates.keys())
    
    def get_template_info(self, name: str) -> Optional[Dict]:
        """获取模板信息"""
        template = self.get_template(name)
        if not template:
            return None
        
        return {
            "name": template.name,
            "description": template.description,
            "variables": template.variables,
            "category": template.category,
            "language": template.language
        }
    
    def validate_template_variables(self, name: str, **kwargs) -> Dict[str, bool]:
        """
        验证模板变量
        
        Args:
            name: 模板名称
            **kwargs: 提供的变量
            
        Returns:
            Dict[str, bool]: 验证结果
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"模板不存在: {name}")
        
        required_vars = set(template.variables)
        provided_vars = set(kwargs.keys())
        
        return {
            "valid": required_vars.issubset(provided_vars),
            "missing_vars": list(required_vars - provided_vars),
            "extra_vars": list(provided_vars - required_vars)
        }
    
    def load_templates_from_dir(self, templates_dir: Union[str, Path]):
        """从目录加载模板"""
        templates_dir = Path(templates_dir)
        if not templates_dir.exists():
            self.logger.warning(f"模板目录不存在: {templates_dir}")
            return
        
        for template_file in templates_dir.glob("*.json"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                
                template = PromptTemplate(**template_data)
                self.register_template(template)
                
            except Exception as e:
                self.logger.error(f"加载模板文件失败 {template_file}: {str(e)}")
    
    def save_template(self, template: PromptTemplate, 
                     output_dir: Union[str, Path]):
        """保存模板到文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        template_file = output_dir / f"{template.name}.json"
        
        try:
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template.dict(), f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"模板已保存: {template_file}")
            
        except Exception as e:
            self.logger.error(f"保存模板失败: {str(e)}")
            raise
    
    def create_custom_template(self, name: str, description: str, 
                             template: str, variables: List[str],
                             category: str = "custom",
                             language: str = "zh") -> PromptTemplate:
        """
        创建自定义模板
        
        Args:
            name: 模板名称
            description: 模板描述
            template: 模板内容
            variables: 模板变量
            category: 模板分类
            language: 语言
            
        Returns:
            PromptTemplate: 创建的模板
        """
        custom_template = PromptTemplate(
            name=name,
            description=description,
            template=template,
            variables=variables,
            category=category,
            language=language
        )
        
        self.register_template(custom_template)
        return custom_template


# 预定义的系统提示词
SYSTEM_PROMPTS = {
    "qa_generator": """你是一个专业的教育内容创作者和问答生成专家。你的任务是基于给定的文本内容，生成高质量、有教育价值的问答对。

核心能力：
1. 深度理解文本内容的核心概念和关键信息
2. 设计多层次、多角度的问题
3. 生成准确、完整的答案
4. 确保问答对具有教育价值

质量标准：
- 问题应该具有启发性，能够促进深度思考
- 答案应该准确、完整，直接来源于原文
- 避免生成过于简单或过于复杂的问题
- 保持问题的多样性和层次性""",
    
    "text_classifier": """你是一个专业的文本分类专家，具有深厚的自然语言处理和机器学习背景。

核心能力：
1. 准确理解文本的语义内容
2. 识别文本的主题、情感、意图等特征
3. 基于上下文进行精准分类
4. 提供分类的置信度和理由

分类原则：
- 仔细分析文本的词汇、句法和语义特征
- 考虑文本的整体语境和隐含意义
- 基于客观证据进行分类判断
- 提供清晰的分类理由""",
    
    "entity_recognizer": """你是一个专业的命名实体识别专家，专长于从文本中准确识别各类命名实体。

核心能力：
1. 识别人名、地名、组织机构名等基础实体
2. 识别时间、数量、专业术语等特殊实体
3. 处理实体边界和嵌套实体问题
4. 理解上下文中的实体语义

识别原则：
- 准确标识实体的起始和结束位置
- 正确分类实体类型
- 处理实体的变体和别名
- 考虑上下文语境的影响""",
    
    "summarizer": """你是一个专业的文本摘要专家，擅长提取和概括文本的核心信息。

核心能力：
1. 识别文本的主要观点和关键信息
2. 生成简洁明了的摘要
3. 保持原文的核心意思不变
4. 适应不同的摘要长度和风格要求

摘要原则：
- 保留最重要的信息和观点
- 保持逻辑结构的清晰性
- 使用简洁准确的语言表达
- 避免添加原文中没有的信息"""
} 