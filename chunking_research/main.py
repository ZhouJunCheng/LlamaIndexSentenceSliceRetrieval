import os
import time
import numpy as np
from typing import List, Dict, Union, Any
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter,TokenTextSplitter,SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.core.schema import Document
from llama_index.core.evaluation import CorrectnessEvaluator

def evaluate_splitter(sentence_splitter, documents, questions: Union[str, List[str]], 
                     ground_truths: Union[str, List[str]], splitter_name: str) -> Dict[str, Any]:
    """
    评估文本分割器在问答任务中的表现，支持单个问题或问题列表
    
    Args:
        sentence_splitter: 文本分割器
        documents: 文档列表
        questions: 单个问题(字符串)或问题列表
        ground_truths: 单个标准答案(字符串)或标准答案列表，与问题一一对应
        splitter_name: 分割器名称
    
    Returns:
        dict: 包含评估指标的字典
    """
    print(f"===== 评估分割器: {splitter_name} =====")
    
    # 确保questions和ground_truths都是列表形式
    if isinstance(questions, str):
        questions = [questions]
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    # 验证问题和答案数量是否匹配
    if len(questions) != len(ground_truths):
        raise ValueError("问题数量与标准答案数量不匹配")
    
    # 记录开始时间
    start_time = time.time()
    
    # 使用分割器处理文档
    nodes = sentence_splitter.get_nodes_from_documents(documents)
    print(f"生成的节点数量: {len(nodes)}")
    
    # 创建索引
    index = VectorStoreIndex(nodes)
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        streaming=True,
        node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")]
    )
    # 创建查询引擎
    # query_engine = index.as_query_engine()
    
    # 存储所有问题的评估结果
    all_scores = []
    all_feedbacks = []
    all_responses = []
    
    # 评估器
    evaluator = CorrectnessEvaluator()
    
    # 对每个问题进行评估
    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
        print(f"问题 {i+1}: {question}")
        
        # 执行查询
        response = query_engine.query(question)
        all_responses.append(str(response))
        
        print(f"回答: {response}")
        print(f"标准答案: {ground_truth}")
        
        # 评估回答的正确性
        eval_result = evaluator.evaluate(
            query=question,
            response=str(response),
            reference=ground_truth
        )
        
        all_scores.append(eval_result.score)
        all_feedbacks.append(eval_result.feedback)
        
        print(f"正确性评分: {eval_result.score}")
        print(f"评估反馈: {eval_result.feedback}")
    
    # 计算总处理时间
    total_processing_time = time.time() - start_time
    
    # 计算平均分数
    avg_score = np.mean(all_scores) if all_scores else 0
    
    # 收集评估指标
    metrics = {
        "splitter_name": splitter_name,
        "total_processing_time": total_processing_time,
        "avg_processing_time": total_processing_time / len(questions),
        "node_count": len(nodes),
        "question_count": len(questions),
        "avg_correctness_score": avg_score,
        "individual_scores": all_scores,
        "individual_feedbacks": all_feedbacks,
        "individual_responses": all_responses
    }
    
    print(f"总体评估结果:")
    print(f"总处理时间: {total_processing_time:.2f}秒")
    print(f"平均每题处理时间: {total_processing_time / len(questions):.2f}秒")
    print(f"平均正确性评分: {avg_score:.2f}")
    
    return metrics

def main():
    """
    主函数：设置环境、加载数据、定义问题和标准答案，并评估不同的文本分割器
    """
    # 在根目录可以通过python -m chunking_research.main 运行
    print("初始化LLM和嵌入模型...")
    
    # 设置LLM模型
    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        is_chat_model=True
    )

    # 设置嵌入模型
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,
        embed_input_length=8192
    )

    print("加载文档...")
    # 加载文档
    documents = SimpleDirectoryReader("data").load_data()
    if not documents:
        print("警告：未找到文档，请确保data目录中有文件")
        return
    
    print(f"成功加载 {len(documents)} 个文档")
    
    # 定义测试问题和标准答案
    questions = [
        "Generous is defined in today’s Merriam Webster as?",
        "At least how many Countries suggest that shorter workweeks?",
        "Why are people who work four days a week more likely to feel exhausted?",
        "Why Hungry waxworms that can eat through plastic and digest it too?"
    ]
    
    # questions = ["蜡虫如何实现塑料降解，其潜在应用与生态风险是什么？"]


    ground_truths = [
        "giving or sharing in abundance and without hesitation.",
        "four",
        "People who work four days a week are more likely to feel exhausted.",
        "Because wax worms secrete saliva containing two key enzymes Ceres and Demeter in their mouths."
    ]
    # ground_truths = ["（聚焦机制：通过唾液酶氧化聚乙烯；应用场景：规模化生产酶制剂；风险提示：释放活体可能破坏蜂巢等生态系统）"]
    
    # 评估不同的文本分割器
    results = []
    
    # 1. 标准句子分割器
    # print("测试标准句子分割器...")
    # sentence_splitter = SentenceSplitter(
    #     chunk_size=512,
    #     chunk_overlap=50
    # )

    # Token 切片
    print("测试标准Token分割器...")
    splitter = TokenTextSplitter(
        chunk_size=128,
        chunk_overlap=4,
        separator="\n"
    )

    # result1 = evaluate_splitter(splitter, documents, questions, ground_truths, "标准冗余分割器chunk_size=512,chunk_overlap=50")
    # results.append(result1)
    
    # 2. 较小块大小的句子分割器
    # print("测试小块大小句子分割器...")
    # small_chunk_splitter = SentenceSplitter(
    #     chunk_size=512,
    #     chunk_overlap=5
    # )

    small_chunk_splitter = TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=4,
        separator="\n"
    )

    # result2 = evaluate_splitter(small_chunk_splitter, documents, questions, ground_truths, "小块句冗余分割器,chunk_size=512,chunk_overlap=5")
    # results.append(result2)
    
    # 3. 较大块大小的句子分割器
    print("测试大块大小句子分割器...")
    # large_chunk_splitter = SentenceSplitter(
    #     chunk_size=512,
    #     chunk_overlap=200
    # )
    large_chunk_splitter = TokenTextSplitter(
        chunk_size=2048,
        chunk_overlap=4,
        separator="\n"
    )

    # result3 = evaluate_splitter(large_chunk_splitter, documents, questions, ground_truths, "大块冗余分割器,chunk_size=512,chunk_overlap=200")
    # results.append(result3)
    
    sentence_window_splitter = SentenceWindowNodeParser.from_defaults(
        window_size=5,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )

    # 注意：句子窗口切片需要特殊的后处理器
    # index = VectorStoreIndex.from_documents(documents)
    
    

    result3 = evaluate_splitter(sentence_window_splitter, documents, questions, ground_truths, "Sentence Window")
    results.append(result3)

    # 比较不同分割器的性能
    print("========== 分割器性能比较 ==========")
    print(f"{'分割器名称':<20} {'平均评分':<10} {'处理时间(秒)':<15} {'节点数量':<10}")
    print("-" * 55)
    
    for result in results:
        print(f"{result['splitter_name']:<20} {result['avg_correctness_score']:<10.2f} {result['total_processing_time']:<15.2f} {result['node_count']:<10}")
    
    # 找出最佳分割器
    best_splitter = max(results, key=lambda x: x['avg_correctness_score'])
    print(f"最佳分割器: {best_splitter['splitter_name']} (平均评分: {best_splitter['avg_correctness_score']:.2f})")


if __name__ == "__main__":
    main()