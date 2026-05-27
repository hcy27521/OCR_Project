import os
import json
from PIL import Image
from tqdm import tqdm
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
from qwen_vl_utils import process_vision_info


import torch

# ======================
# 使用第二张4090
# ======================

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ======================
# 1. 加载模型
# ======================
MODEL_PATH = "./Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

# ======================
# 2. 数据路径
# ======================

DATASET_DIR = "dataset"
OUTPUT_JSON = "output/result.json"

# 保存结果
results = {}

# ======================
# 3. 遍历图片
# ======================

image_files = [
    f for f in os.listdir(DATASET_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

for image_name in tqdm(image_files):

    image_path = os.path.join(DATASET_DIR, image_name)

    # 构造消息
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
                        "请识别图片中的全部内容。"
                        "数学公式请转换为 LaTeX 格式。"
                        "只输出识别结果，不要解释。"
                    )
                },
            ],
        }
    ]

    # 处理输入
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # 推理
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # 保存
    results[image_name] = output_text

# ======================
# 4. 保存JSON
# ======================

os.makedirs("output", exist_ok=True)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("OCR识别完成！")
print(f"结果已保存到：{OUTPUT_JSON}")