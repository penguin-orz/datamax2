"""
语料标注模块演示脚本

展示DataMax语料标注模块的完整功能，包括：
- 文本分割
- QA生成
- 数据导出
- 批量处理

使用方法:
    python -m datamax.labeler.demo
"""

import json
from pathlib import Path
from typing import List

from loguru import logger

from .base_labeler import LabelConfig
from .llm_client import BaseLLMClient
from .prompt_manager import PromptManager
from .qa_labeler import QALabeler, QAConfig
from .text_splitter import TextSplitter, SplitterConfig
from .token_counter import TokenCounter


class DemoLLMClient(BaseLLMClient):
    """演示用的模拟LLM客户端"""
    
    def __init__(self):
        self.call_count = 0
    
    def chat(self, messages):
        """模拟对话响应"""
        self.call_count += 1
        
        # 模拟不同类型的响应
        if self.call_count % 3 == 1:
            return """
            1. 问题：什么是人工智能？
               答案：人工智能（AI）是一门研究如何让机器具备类似人类智能的学科，包括学习、推理、感知等能力。
               
            2. 问题：机器学习的主要作用是什么？
               答案：机器学习是AI的核心技术之一，它让计算机能够从数据中自动学习规律，而无需明确编程。
               
            3. 问题：深度学习有什么特点？
               答案：深度学习使用多层神经网络来解决复杂问题，在图像识别和自然语言处理方面取得了突破性进展。
            """
        elif self.call_count % 3 == 2:
            return """
            1. 问题：NLP技术的应用领域有哪些？
               答案：NLP技术让计算机能够理解和生成人类语言，广泛应用于搜索引擎、翻译软件、智能客服等领域。
               
            2. 问题：人工智能包含哪些子领域？
               答案：人工智能包括机器学习、深度学习、自然语言处理、计算机视觉、专家系统等多个重要子领域。
            """
        else:
            return """
            1. 问题：什么是神经网络？
               答案：神经网络是模拟人脑神经元连接方式的计算模型，由多个互联的节点组成，能够学习复杂的数据模式。
               
            2. 问题：数据在机器学习中的重要性如何？
               答案：数据是机器学习的基础，高质量的数据能帮助模型学习到更准确的规律，是AI系统成功的关键因素。
            """
    
    async def achat(self, messages):
        """异步对话响应"""
        return self.chat(messages)
    
    def count_tokens(self, text: str) -> int:
        """估算token数量"""
        return len(text) // 3


def create_sample_documents():
    """创建示例文档"""
    documents = [
        {
            "title": "人工智能基础",
            "content": """
            人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，旨在创建能够模拟人类智能的机器和软件系统。
            
            ## 发展历程
            人工智能的概念最早可以追溯到1950年代，当时计算机科学家开始探索机器是否能够"思考"。
            经过几十年的发展，AI技术已经在各个领域取得了显著进展。
            
            ## 主要技术
            现代AI主要包括机器学习、深度学习、自然语言处理、计算机视觉等技术分支。
            这些技术的结合使得AI系统能够处理越来越复杂的任务。
            
            ## 应用前景
            AI技术正在改变我们的生活和工作方式，从智能手机到自动驾驶汽车，
            从医疗诊断到金融分析，AI的应用前景十分广阔。
            """
        },
        {
            "title": "机器学习详解", 
            "content": """
            机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下从数据中学习。
            
            ## 学习方式
            机器学习主要分为三种学习方式：
            1. 监督学习：使用标记数据进行训练
            2. 无监督学习：从未标记数据中发现模式
            3. 强化学习：通过与环境交互学习最优策略
            
            ## 算法类型
            常见的机器学习算法包括线性回归、决策树、随机森林、支持向量机、神经网络等。
            不同的算法适用于不同类型的问题和数据。
            
            ## 实际应用
            机器学习在推荐系统、图像识别、语音识别、自然语言处理等领域都有广泛应用。
            """
        },
        {
            "title": "深度学习革命",
            "content": """
            深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的学习过程。
            
            ## 技术原理
            深度学习通过构建多层的人工神经网络，每一层都能提取和转换输入数据的特征。
            这种层次化的特征学习使得深度学习能够处理非常复杂的数据模式。
            
            ## 突破性进展
            深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展，
            在某些任务上甚至超越了人类的表现。
            
            ## 计算要求
            深度学习通常需要大量的计算资源和数据，GPU和专用芯片的发展为深度学习提供了强大的计算支持。
            """
        }
    ]
    
    return documents


def demo_text_splitting():
    """演示文本分割功能"""
    logger.info("🔪 开始演示文本分割功能...")
    
    # 创建分割器配置
    config = SplitterConfig(
        chunk_size=300,
        chunk_overlap=50,
        keep_separator=True
    )
    
    # 测试递归分割器
    recursive_splitter = TextSplitter.create_splitter("recursive", config)
    
    sample_text = """
    人工智能是计算机科学的一个重要分支。它旨在创建能够模拟人类智能的机器。
    
    机器学习是AI的核心技术。它让计算机能够从数据中自动学习规律。
    通过训练，机器学习模型可以对新数据进行预测和分类。
    
    深度学习是机器学习的一个子领域。它使用多层神经网络来解决复杂问题。
    在图像识别和自然语言处理方面，深度学习取得了突破性进展。
    """
    
    chunks = recursive_splitter.split_text(sample_text)
    logger.info(f"递归分割器生成了 {len(chunks)} 个文本块")
    
    for i, chunk in enumerate(chunks):
        logger.info(f"块 {i+1}: {chunk.content[:50]}...")
    
    # 测试语义分割器
    semantic_splitter = TextSplitter.create_splitter("semantic", config)
    semantic_chunks = semantic_splitter.split_text(sample_text)
    logger.info(f"语义分割器生成了 {len(semantic_chunks)} 个文本块")


def demo_qa_generation():
    """演示QA生成功能"""
    logger.info("🤖 开始演示QA生成功能...")
    
    # 创建模拟LLM客户端
    llm_client = DemoLLMClient()
    
    # 创建QA配置
    qa_config = QAConfig(
        num_qa_per_chunk=3,
        question_types=["factual", "comprehension", "application"],
        difficulty_levels=["easy", "medium", "hard"],
        enable_filtering=True
    )
    
    # 创建QA标注器
    qa_labeler = QALabeler(llm_client, qa_config)
    
    # 生成QA对
    sample_texts = [
        "人工智能是一门研究如何让机器具备类似人类智能的学科。它包括机器学习、深度学习、自然语言处理等多个子领域。",
        "机器学习让计算机能够从数据中自动学习规律，无需明确编程。它是现代AI系统的核心技术之一。"
    ]
    
    all_results = []
    for i, text in enumerate(sample_texts):
        logger.info(f"处理文本 {i+1}...")
        results = qa_labeler.label_text(text)
        all_results.extend(results)
        
        for result in results:
            logger.info(f"成功生成 {len(result.qa_pairs)} 个QA对")
    
    return all_results


def demo_batch_processing():
    """演示批量处理功能"""
    logger.info("📦 开始演示批量处理功能...")
    
    # 创建示例文档
    documents = create_sample_documents()
    
    # 保存为临时文件
    temp_files = []
    for i, doc in enumerate(documents):
        file_path = Path(f"temp_doc_{i+1}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# {doc['title']}\n\n{doc['content']}")
        temp_files.append(file_path)
    
    try:
        # 创建QA标注器
        llm_client = DemoLLMClient()
        qa_labeler = QALabeler(llm_client)
        
        # 批量处理文件
        all_results = []
        for file_path in temp_files:
            logger.info(f"处理文件: {file_path}")
            results = qa_labeler.generate_qa_from_file(str(file_path))
            all_results.extend(results)
            logger.info(f"从文件生成了 {len(results)} 个QA结果")
        
        return all_results
        
    finally:
        # 清理临时文件
        for file_path in temp_files:
            if file_path.exists():
                file_path.unlink()


def demo_data_export():
    """演示数据导出功能"""
    logger.info("💾 开始演示数据导出功能...")
    
    # 生成一些QA数据
    llm_client = DemoLLMClient()
    qa_labeler = QALabeler(llm_client)
    
    sample_text = """
    深度学习是机器学习的一个重要分支，它使用多层神经网络来模拟人脑的学习过程。
    通过层次化的特征学习，深度学习能够处理非常复杂的数据模式，
    在图像识别、语音识别、自然语言处理等领域都取得了突破性进展。
    """
    
    results = qa_labeler.label_text(sample_text)
    
    # 导出为不同格式
    formats = ["jsonl", "json", "csv"]
    exported_files = []
    
    for format_type in formats:
        output_file = f"demo_output.{format_type}"
        try:
            qa_labeler.export_qa_dataset(results, output_file, format_type)
            exported_files.append(output_file)
            logger.info(f"成功导出为 {format_type.upper()} 格式: {output_file}")
            
            # 展示导出内容的预览
            if format_type == "jsonl":
                with open(output_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    logger.info(f"JSONL文件包含 {len(lines)} 行数据")
                    if lines:
                        sample_data = json.loads(lines[0])
                        logger.info(f"示例QA对: {sample_data['question'][:50]}...")
                        
        except Exception as e:
            logger.error(f"导出 {format_type} 格式失败: {str(e)}")
    
    return exported_files


def demo_statistics():
    """演示统计功能"""
    logger.info("📊 开始演示统计功能...")
    
    # 创建token计数器
    token_counter = TokenCounter()
    
    # 测试文本
    test_texts = [
        "人工智能是计算机科学的一个分支。",
        "机器学习让计算机能够从数据中学习。",
        "深度学习使用多层神经网络处理复杂问题。"
    ]
    
    total_tokens = 0
    for i, text in enumerate(test_texts):
        token_count = token_counter.count_tokens(text)
        total_tokens += token_count
        logger.info(f"文本 {i+1} 包含 {token_count} 个tokens")
    
    logger.info(f"总共 {total_tokens} 个tokens")
    
    # 演示统计信息
    avg_tokens = total_tokens / len(test_texts)
    logger.info(f"平均每个文本 {avg_tokens:.1f} tokens")


def main():
    """运行完整演示"""
    logger.info("🚀 DataMax语料标注模块演示开始...")
    
    try:
        # 1. 文本分割演示
        demo_text_splitting()
        
        # 2. QA生成演示
        qa_results = demo_qa_generation()
        
        # 3. 批量处理演示
        batch_results = demo_batch_processing()
        
        # 4. 数据导出演示
        exported_files = demo_data_export()
        
        # 5. 统计功能演示
        demo_statistics()
        
        # 总结
        logger.info("📈 演示总结:")
        logger.info(f"- 生成了 {len(qa_results)} 个QA结果（单独处理）")
        logger.info(f"- 生成了 {len(batch_results)} 个QA结果（批量处理）")
        logger.info(f"- 导出了 {len(exported_files)} 个文件")
        
        logger.info("🎉 DataMax语料标注模块演示完成！")
        
        # 清理演示文件
        cleanup_files = ["demo_output.jsonl", "demo_output.json", "demo_output.csv"]
        for file_path in cleanup_files:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"清理演示文件: {file_path}")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main() 