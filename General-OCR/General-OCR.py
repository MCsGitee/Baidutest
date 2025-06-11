from paddleocr import PaddleOCR
import os
import json

# 初始化OCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="ch",
    layout=True,  # 启用布局分析
    use_gpu=False
)

# 确保输出目录存在
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 执行OCR
image_path = "general_ocr_002.png"
result = ocr.ocr(image_path, cls=True)

# 保存JSON结果
json_path = os.path.join(output_dir, "result.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# 保存可视化结果（需要额外处理）
from PIL import Image, ImageDraw

img = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(img)

for res in result:
    for line in res:
        box = line[0]
        # 绘制文本框
        draw.polygon([(int(p[0]), int(p[1])) for p in box], outline=(255, 0, 0))

# 保存可视化图片
img.save(os.path.join(output_dir, "visualization.jpg"))

print(f"结果已保存到 {output_dir} 目录")
