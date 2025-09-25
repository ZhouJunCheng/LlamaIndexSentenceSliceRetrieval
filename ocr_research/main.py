

import os
import json
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

from imageOCRReader import ImageOCRReader

def main():
# 作业的入口写在这里。你可以就写这个文件，或者扩展多个文件，但是执行入口留在这里。
# 在根目录可以通过python -m ocr_research.main 运行
    
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
     
    reader = ImageOCRReader(lang='ch', use_gpu=False)
    # 加载文档
    documents = reader.load_data(["general_ocr_002.png", "images/202503月.png", "images/202504月.png","images/202505月.png"])
    if not documents:
        print("警告：未找到文档，请确保data目录中有文件")
        return
    print(f"成功加载 {len(documents)} 个文档")
    
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    response = query_engine.query("图片中3月电费总额是多少？")
    print("图片中3月电费总额是多少？")
    print(response)

if __name__ == "__main__":
    main()