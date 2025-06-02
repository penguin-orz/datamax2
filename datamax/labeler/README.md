# DataMax 语料标注模块

DataMax语料标注模块是一个强大的智能语料处理工具，基于大语言模型（LLM）提供高质量的QA对生成、文本分割、数据导出等功能。

## 🌟 核心功能

### 1. 智能QA对生成
- 基于LLM的高质量问答对生成
- 支持多种问题类型（事实性、理解性、应用性）
- 可配置难度级别（简单、中等、困难）
- 自动质量过滤和验证

### 2. 文本分割策略
- **递归分割器**：基于分隔符的层次化分割
- **语义分割器**：基于段落和语义边界的智能分割
- 支持自定义分割参数（块大小、重叠度等）

### 3. 精确Token计数
- 支持tiktoken精确计数
- 多种回退估算方法
- 成本预估功能
- 按token数量分割文本

### 4. 灵活的提示词管理
- 内置多种任务模板
- 支持自定义提示词
- 模板参数化和动态替换

### 5. 多格式数据导出
- JSONL格式（适合大规模数据处理）
- JSON格式（适合小规模数据和调试）
- CSV格式（适合数据分析和可视化）

## 🚀 快速开始

### 安装依赖

```bash
pip install tiktoken unstructured langchain langchain-community pydantic loguru
```

### 基本使用

```python
from datamax.labeler import QALabeler, TextSplitter, TokenCounter

# 1. 创建LLM客户端（需要实现BaseLLMClient接口）
llm_client = YourLLMClient()

# 2. 创建QA标注器
qa_labeler = QALabeler(llm_client)

# 3. 生成QA对
text = "人工智能是计算机科学的一个重要分支，旨在创建能够模拟人类智能的机器。"
results = qa_labeler.label_text(text)

# 4. 导出数据
qa_labeler.export_qa_dataset(results, "output.jsonl", "jsonl")
```

### 文本分割示例

```python
from datamax.labeler import TextSplitter, SplitterConfig

# 创建分割器配置
config = SplitterConfig(
    chunk_size=500,
    chunk_overlap=100
)

# 创建递归分割器
splitter = TextSplitter.create_splitter("recursive", config)

# 分割文本
chunks = splitter.split_text(long_text)

# 分割文件
file_chunks = splitter.split_file("document.txt")
```

### Token计数示例

```python
from datamax.labeler import TokenCounter

# 创建token计数器
counter = TokenCounter()

# 计算token数量
token_count = counter.count_tokens("这是一段测试文本")

# 批量计算
texts = ["文本1", "文本2", "文本3"]
token_counts = counter.count_tokens_batch(texts)

# 获取统计信息
stats = counter.get_token_distribution(texts)

# 估算成本
cost_info = counter.estimate_cost("输入文本")
```

## 📖 详细配置

### QA标注器配置

```python
from datamax.labeler import QAConfig

config = QAConfig(
    num_qa_per_chunk=3,           # 每个文本块生成的QA对数量
    question_types=[              # 问题类型列表
        "factual",               # 事实性问题
        "comprehension",         # 理解性问题
        "application"            # 应用性问题
    ],
    difficulty_levels=[           # 难度级别
        "easy", "medium", "hard"
    ],
    min_answer_length=10,         # 答案最小长度
    max_answer_length=500,        # 答案最大长度
    enable_filtering=True         # 启用质量过滤
)

qa_labeler = QALabeler(llm_client, config)
```

### 文本分割器配置

```python
from datamax.labeler import SplitterConfig

config = SplitterConfig(
    chunk_size=1000,              # 块大小
    chunk_overlap=200,            # 重叠大小
    separators=["\n\n", "\n", " ", ""],  # 分隔符列表
    keep_separator=True           # 是否保留分隔符
)
```

### Token计数器配置

```python
from datamax.labeler import TokenCounterConfig

config = TokenCounterConfig(
    encoding_name="cl100k_base",  # 编码名称
    model_name="gpt-3.5-turbo",  # 模型名称（可选）
    fallback_method="estimate"    # 回退方法
)

counter = TokenCounter(config)
```

## 🔧 自定义LLM客户端

要使用自己的LLM服务，需要实现`BaseLLMClient`接口：

```python
from datamax.labeler import BaseLLMClient

class MyLLMClient(BaseLLMClient):
    def chat(self, messages):
        # 实现同步对话接口
        # messages: 对话消息列表
        # 返回: 字符串响应
        pass
    
    async def achat(self, messages):
        # 实现异步对话接口
        # 可以简单调用同步方法
        return self.chat(messages)
    
    def count_tokens(self, text: str) -> int:
        # 实现token计数接口
        # 返回: token数量
        pass
```

## 📊 数据导出格式

### JSONL格式
```json
{"question": "什么是人工智能？", "answer": "人工智能是...", "type": "factual", "difficulty": "easy"}
{"question": "机器学习的原理是什么？", "answer": "机器学习是...", "type": "comprehension", "difficulty": "medium"}
```

### JSON格式
```json
[
  {
    "question": "什么是人工智能？",
    "answer": "人工智能是...",
    "type": "factual",
    "difficulty": "easy",
    "source_text": "人工智能是计算机科学的一个重要分支...",
    "timestamp": "2024-01-01T12:00:00"
  }
]
```

### CSV格式
| question | answer | type | difficulty | source_text | timestamp |
|----------|--------|------|------------|-------------|-----------|
| 什么是人工智能？ | 人工智能是... | factual | easy | 人工智能是计算机... | 2024-01-01T12:00:00 |

## 🧪 运行测试

```bash
# 运行完整测试
python -m datamax.labeler.test_labeler

# 运行演示
python -m datamax.labeler.demo
```

## 📝 注意事项

1. **LLM客户端**：必须实现`BaseLLMClient`接口才能使用QA生成功能
2. **依赖包**：确保安装了所有必需的依赖包
3. **文件编码**：建议使用UTF-8编码处理中文文本
4. **内存管理**：处理大文件时注意内存使用情况
5. **API配额**：使用外部LLM服务时注意API调用限制

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个模块！

## 📄 许可证

本项目采用MIT许可证。 