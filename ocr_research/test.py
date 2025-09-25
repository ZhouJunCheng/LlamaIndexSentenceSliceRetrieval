from paddleocr import PaddleOCR

# 初始化PaddleOCR
ocr = PaddleOCR()

# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_server_det",
#     text_recognition_model_name="PP-OCRv5_server_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# )  # 更换 PP-OCRv5_server 模型

# 确保使用正确的图像路径
import os
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建图像的绝对路径
image_path = os.path.join(current_dir, "general_ocr_002.png")
# 检查文件是否存在
if not os.path.exists(image_path):
    print(f"警告：图像文件 {image_path} 不存在！")
    # 尝试在images目录下查找
    image_path = os.path.join(current_dir, "images", "general_ocr_002.png")
    if not os.path.exists(image_path):
        print(f"警告：图像文件 {image_path} 也不存在！")
        # 列出images目录下的文件
        images_dir = os.path.join(current_dir, "images")
        if os.path.exists(images_dir):
            print(f"images目录下的文件：{os.listdir(images_dir)}")
            # 使用第一个图像文件
            files = os.listdir(images_dir)
            if files:
                image_path = os.path.join(images_dir, files[0])
                print(f"将使用第一个可用的图像文件：{image_path}")

# 使用OCR识别图像
print(f"正在处理图像：{image_path}")
result = ocr.ocr(image_path, cls=False)  # 使用ocr方法而不是predict

# 处理结果
if result:
    for idx, line in enumerate(result):
        if line:
            print(f"第 {idx+1} 行结果：")
            for item in line:
                print(f"位置：{item[0]}, 文本：{item[1][0]}, 置信度：{item[1][1]}")
            
            # 保存结果到output目录
            output_dir = os.path.join(current_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 注意：PaddleOCR的新版本可能没有直接的save_to_img和save_to_json方法
            # 如果这些方法不可用，我们需要手动保存结果
            try:
                # 尝试使用可能存在的方法
                line.save_to_img(output_dir)
                line.save_to_json(output_dir)
            except (AttributeError, TypeError):
                print("注意：无法使用save_to_img和save_to_json方法，这可能是因为PaddleOCR版本差异")
else:
    print("OCR识别未返回结果")
