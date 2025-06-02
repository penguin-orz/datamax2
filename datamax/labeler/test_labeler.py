"""
语料标注模块测试

测试所有labeler模块的功能，确保代码可以正常运行
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
from loguru import logger

from datamax.labeler import (
    BaseLabeler, LabelConfig, LabelResult,
    LLMClient, LLMConfig, TokenCounter,
    PromptManager, PromptTemplate,
    QALabeler, QAConfig, QAResult,
    TextSplitter, SplitterConfig
)


class MockLLMClient:
    """模拟LLM客户端，用于测试"""
    
    def __init__(self):
        self.logger = logger
    
    def chat(self, messages):
        """模拟聊天响应"""
        from datamax.labeler.llm_client import LLMResponse
        
        # 根据消息内容返回不同的模拟响应
        user_message = messages[-1].content if messages else ""
        
        if "QA" in user_message or "问答" in user_message:
            mock_response = {
                "qa_pairs": [
                    {
                        "question": "什么是人工智能？",
                        "answer": "人工智能是一门研究如何让机器模拟人类智能的科学。",
                        "type": "factual"
                    },
                    {
                        "question": "机器学习的核心思想是什么？",
                        "answer": "机器学习的核心思想是让计算机通过数据学习规律，无需明确编程。",
                        "type": "comprehension"
                    }
                ]
            }
            content = f"```json\n{json.dumps(mock_response, ensure_ascii=False)}\n```"
        else:
            content = "这是一个模拟的LLM响应。"
        
        return LLMResponse(
            content=content,
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="mock-model",
            finish_reason="stop",
            response_time=1.0
        )
    
    def count_tokens(self, text: str) -> int:
        """模拟token计数"""
        return len(text) // 4


def test_text_splitter():
    """测试文本分割器"""
    logger.info("开始测试文本分割器...")
    
    # 测试基本分割
    text = """
    人工智能（Artificial Intelligence，简称AI）是一门研究如何让机器模拟人类智能的科学。
    它包括机器学习、深度学习、自然语言处理等多个分支领域。
    
    机器学习是AI的重要组成部分，其核心思想是让计算机通过数据学习规律，而无需明确编程。
    深度学习则是机器学习的一个子集，它使用神经网络来解决复杂问题。
    
    自然语言处理（NLP）专注于让计算机理解和生成人类语言。
    它在搜索引擎、机器翻译、聊天机器人等应用中发挥重要作用。
    """
    
    # 测试递归分割器
    config = SplitterConfig(chunk_size=200, chunk_overlap=50)
    splitter = TextSplitter.create_splitter("recursive", config)
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 0, "应该生成至少一个文本块"
    assert all(len(chunk.content) <= 250 for chunk in chunks), "所有块的大小应该在限制范围内"
    
    logger.info(f"递归分割器测试通过，生成了 {len(chunks)} 个文本块")
    
    # 测试语义分割器
    semantic_splitter = TextSplitter.create_splitter("semantic", config)
    semantic_chunks = semantic_splitter.split_text(text)
    
    assert len(semantic_chunks) > 0, "语义分割器应该生成至少一个文本块"
    
    logger.info(f"语义分割器测试通过，生成了 {len(semantic_chunks)} 个文本块")
    
    # 测试快速分割
    quick_chunks = TextSplitter.quick_split(text, chunk_size=150)
    assert len(quick_chunks) > 0, "快速分割应该生成至少一个文本块"
    
    logger.info("文本分割器测试完成 ✓")


def test_prompt_manager():
    """测试提示词管理器"""
    logger.info("开始测试提示词管理器...")
    
    # 创建提示词管理器
    prompt_manager = PromptManager()
    
    # 测试获取默认模板
    qa_template = prompt_manager.get_template("qa_generation")
    assert qa_template is not None, "应该能获取QA生成模板"
    assert "text" in qa_template.variables, "QA模板应该包含text变量"
    
    # 测试渲染模板
    rendered = prompt_manager.render_template(
        "qa_generation",
        text="测试文本",
        num_qa=3
    )
    assert "测试文本" in rendered, "渲染后的文本应该包含输入文本"
    assert "3" in rendered, "渲染后的文本应该包含QA数量"
    
    # 测试列出模板
    templates = prompt_manager.list_templates()
    assert len(templates) > 0, "应该有预定义的模板"
    
    qa_templates = prompt_manager.list_templates(category="qa")
    assert len(qa_templates) > 0, "应该有QA类别的模板"
    
    # 测试创建自定义模板
    custom_template = prompt_manager.create_custom_template(
        name="test_template",
        description="测试模板",
        template="这是一个测试模板：{{test_var}}",
        variables=["test_var"],
        category="test"
    )
    
    assert custom_template.name == "test_template", "自定义模板名称应该正确"
    
    # 测试验证模板变量
    validation = prompt_manager.validate_template_variables(
        "test_template",
        test_var="测试值"
    )
    assert validation["valid"] is True, "提供了所需变量，验证应该通过"
    
    logger.info("提示词管理器测试完成 ✓")


def test_token_counter():
    """测试token计数器"""
    logger.info("开始测试token计数器...")
    
    counter = TokenCounter()
    
    # 测试基本计数
    text = "这是一段测试文本，用来验证token计数功能。"
    token_count = counter.count_tokens(text)
    assert token_count > 0, "token数量应该大于0"
    
    # 测试消息计数
    from datamax.labeler.llm_client import ChatMessage
    messages = [
        ChatMessage(role="system", content="你是一个助手"),
        ChatMessage(role="user", content="请回答我的问题"),
        ChatMessage(role="assistant", content="好的，我会帮助你")
    ]
    
    total_tokens = counter.count_messages_tokens(messages)
    assert total_tokens > 0, "消息总token数应该大于0"
    
    # 测试成本估算
    cost_info = counter.estimate_cost(100, 50, "gpt-3.5-turbo")
    assert "total_cost" in cost_info, "成本信息应该包含总成本"
    assert cost_info["total_cost"] > 0, "总成本应该大于0"
    
    logger.info("token计数器测试完成 ✓")


def test_qa_labeler():
    """测试QA标注器"""
    logger.info("开始测试QA标注器...")
    
    # 创建模拟LLM客户端
    mock_client = MockLLMClient()
    
    # 创建QA配置
    config = QAConfig(
        chunk_size=300,
        chunk_overlap=50,
        num_qa_per_chunk=2,
        enable_filtering=True
    )
    
    # 创建QA标注器
    qa_labeler = QALabeler(mock_client, config)
    
    # 测试文本
    test_text = """
    人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。
    这些任务包括学习、推理、问题解决、感知和语言理解。
    
    机器学习是人工智能的一个重要子领域，它使计算机能够从数据中学习，而无需明确编程。
    深度学习进一步扩展了这一概念，使用神经网络来解决更复杂的问题。
    """
    
    # 测试文本标注
    results = qa_labeler.label_text(test_text)
    assert len(results) > 0, "应该生成至少一个QA结果"
    
    for result in results:
        assert isinstance(result, QAResult), "结果应该是QAResult类型"
        assert len(result.qa_pairs) > 0, "应该生成至少一个QA对"
        
        for qa_pair in result.qa_pairs:
            assert qa_pair.question.strip(), "问题不应该为空"
            assert qa_pair.answer.strip(), "答案不应该为空"
    
    logger.info(f"QA标注测试通过，生成了 {len(results)} 个结果")
    
    # 测试批量处理
    texts = [test_text, "这是另一段测试文本。"]
    batch_results = qa_labeler.batch_label_texts(texts)
    assert len(batch_results) == len(texts), "批量结果数量应该与输入文本数量一致"
    
    logger.info("QA标注器测试完成 ✓")


def test_integration():
    """集成测试"""
    logger.info("开始集成测试...")
    
    # 创建临时测试文件
    test_content = """
    # 人工智能基础知识
    
    人工智能（AI）是一门研究如何让机器具备类似人类智能的学科。
    它包括多个重要的子领域：
    
    ## 机器学习
    机器学习是AI的核心技术之一，它让计算机能够从数据中自动学习规律。
    
    ## 深度学习
    深度学习使用多层神经网络来解决复杂问题，在图像识别和自然语言处理方面取得了突破性进展。
    
    ## 自然语言处理
    NLP技术让计算机能够理解和生成人类语言，广泛应用于搜索引擎、翻译软件等。
    """
    
    test_file = Path("test_document.md")
    try:
        # 写入测试文件
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # 创建模拟LLM客户端和QA标注器
        mock_client = MockLLMClient()
        qa_labeler = QALabeler(mock_client)
        
        # 从文件生成QA对
        results = qa_labeler.generate_qa_from_file(str(test_file))
        assert len(results) > 0, "应该从文件生成QA对"
        
        # 导出测试
        output_file = Path("test_qa_output.jsonl")
        try:
            qa_labeler.export_qa_dataset(results, str(output_file), "jsonl")
            assert output_file.exists(), "输出文件应该存在"
            
            # 验证输出内容
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) > 0, "输出文件应该有内容"
                
                # 验证每行都是有效的JSON
                for line in lines:
                    qa_data = json.loads(line.strip())
                    assert "question" in qa_data, "每个QA对应该包含问题"
                    assert "answer" in qa_data, "每个QA对应该包含答案"
            
            logger.info(f"成功导出 {len(lines)} 个QA对到文件")
            
        finally:
            # 清理输出文件
            if output_file.exists():
                output_file.unlink()
    
    finally:
        # 清理测试文件
        if test_file.exists():
            test_file.unlink()
    
    logger.info("集成测试完成 ✓")


def main():
    """运行所有测试"""
    logger.info("🚀 开始运行语料标注模块测试...")
    
    try:
        # 运行各个测试
        test_text_splitter()
        test_prompt_manager()
        test_token_counter()
        test_qa_labeler()
        test_integration()
        
        logger.info("🎉 所有测试通过！语料标注模块功能正常。")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        raise


if __name__ == "__main__":
    main() 