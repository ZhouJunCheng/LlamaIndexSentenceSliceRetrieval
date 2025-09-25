from llama_index.core.readers.base import BaseReader
from paddleocr import PaddleOCR
from llama_index.core.schema import Document
import os
from typing import List, Union

class ImageOCRReader(BaseReader):
    """使用 PP-OCR v5 从图像中提取文本并返回 Document"""
    
    def __init__(self, lang='ch', use_gpu=False, **kwargs):
        """
        Args:
            lang: OCR 语言 ('ch', 'en', 'fr', etc.)
            use_gpu: 是否使用 GPU 加速
            **kwargs: 其他传递给 PaddleOCR 的参数
        """
        self.lang = lang
        self.device = "cpu"
        if use_gpu:
            self.device = "gpu:0"
        self.ocr_reader = PaddleOCR(lang=lang, **kwargs)
        self.ocr_model = "PP-OCRv5"
    def load_data(self, file: Union[str, List[str]]) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表
        Args:
            file: 图像路径字符串 或 路径列表
        Returns:
            List[Document]
        """
        # 实现 OCR 提取逻辑
        # 将每张图的识别结果拼接成文本
        # 构造 Document 对象，附带元数据（如 image_path, ocr_confidence_avg）
        if isinstance(file, str):
            file_list = [file]
        else:
            file_list = file

        documents: List[Document] = []

        for ocr_img_file_url in file:

            if not os.path.exists(ocr_img_file_url):
                print(f"警告：文件不存在，跳过 -> {ocr_img_file_url}")
                continue
            # 执行OCR识别
            ocr_result = self.ocr_reader.ocr(ocr_img_file_url)
            
            # 提取文本和置信度
            page_text_list: List[str] = []
            confidence_list: List[float] = []
            num_text_blocks = 0
            if ocr_result and isinstance(ocr_result[0], dict):
                result_dict = ocr_result[0]
                rec_texts = result_dict.get('rec_texts', [])
                num_text_blocks = len(rec_texts)
                rec_scores = result_dict.get('rec_scores', [])

                for i in range(len(rec_texts)):
                    rec_text = rec_texts[i]
                    confidence = rec_scores[i]
                    page_text_list.append(rec_text)
                    confidence_list.append(confidence)
            
            # 拼接文本
            full_text = ' '.join(page_text_list)
            
            # 计算平均置信度
            avg_confidence  = sum(confidence_list) / len(confidence_list) if confidence_list  else 0.0
            
            # 构造Document对象，附带元数据
            metadata = {
                "image_path": ocr_img_file_url,
                "ocr_confidence_avg": round(avg_confidence,2),
                "language": self.lang,
                "ocr_model": self.ocr_model,
                "num_text_blocks": num_text_blocks,
            }

            doc = Document(
                text=full_text,
                metadata=metadata,
                extra_info={"file_path": ocr_img_file_url}
            )
            documents.append(doc)
        return documents

