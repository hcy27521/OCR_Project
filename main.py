import os
import json
from tqdm import tqdm

import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor  # 自动处理器，用于图像和文本的预处理
)

from qwen_vl_utils import process_vision_info

# =====================================================
# GPU优化
# =====================================================

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# =====================================================
# 模型路径
# =====================================================

MODEL_PATH = "./Qwen2.5-VL-3B-Instruct"

# =====================================================
# 加载模型
# =====================================================

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

# 加载处理器：负责图像尺寸调整、文本分词、模板应用等
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("模型加载成功！")

# =====================================================
# 数据路径
# =====================================================

DATASET_DIR = "dataset"
OUTPUT_JSON = "output/result.json"

# =====================================================
# 自动断点续跑
# =====================================================
# 如果输出 JSON 文件已经存在，则加载已有结果，实现中断后继续处理（不重复识别已完成的图片）
if os.path.exists(OUTPUT_JSON):

    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"检测到已有结果：{len(results)} 条")

else:

    results = {}

# =====================================================
# 获取图片列表
# =====================================================

image_files = sorted([
    f for f in os.listdir(DATASET_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

print(f"共检测到 {len(image_files)} 张图片")

# =====================================================
# OCR识别
# =====================================================

for idx, image_name in enumerate(tqdm(image_files)):

    # 跳过已经识别的图片
    if image_name in results:
        continue

    try:

        image_path = os.path.join(DATASET_DIR, image_name)

        # =================================================
        # 构造Prompt
        # =================================================

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": (
                            "你是OCR文字识别系统。"
                            "请逐字识别图片中的所有文字内容。"
                            "不要总结。"
                            "不要推理。"
                            "不要补充答案。"
                            "不要猜测缺失内容。"
                            "保持原题格式。"
                            "数学公式转换为LaTeX格式。"
                            "只输出图片中真实存在的文本。"
                        )
                    },
                    
                ],
            }
        ]

        # =================================================
        # 处理输入
        # =================================================
        # 将 messages 格式化为模型所需的文本字符串（包括特殊标记）
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 从 messages 中提取图像/视频的底层数据（如像素值、尺寸等），供 processor 使用
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 放到模型所在设备
        inputs = inputs.to(model.device)

        # =================================================
        # 模型推理
        # =================================================

        with torch.no_grad():

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0
            )

        # =================================================
        # 解码输出
        # =================================================

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # =================================================
        # 文本清理
        # =================================================

        output_text = output_text.replace("\n", " ")
        output_text = " ".join(output_text.split())

        # 保存结果
        results[image_name] = output_text

        # =================================================
        # 每10张自动保存
        # =================================================

        if len(results) % 10 == 0:

            os.makedirs("output", exist_ok=True)

            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            print(f"已保存 {len(results)} 条结果")

    except Exception as e:

        print(f"{image_name} 出错: {e}")

        # 错误记录
        results[image_name] = "ERROR"

        # 出错也保存
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        continue

# =====================================================
# 最终保存
# =====================================================

os.makedirs("output", exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("OCR识别完成！")
print(f"结果已保存到：{OUTPUT_JSON}")