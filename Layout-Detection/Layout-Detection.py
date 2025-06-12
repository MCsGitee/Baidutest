"""
from paddleocr import PPStructure
import cv2
import os
import json
import numpy as np
from PIL import ImageFont, ImageDraw, Image


class NumpyEncoder(json.JSONEncoder):


    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)


def analyze_layout(img_path):
    # 初始化模型
    table_engine = PPStructure(
        layout_model_name='PP-DocLayoutV2',
        show_log=True,
        use_gpu=True
    )

    os.makedirs('./output', exist_ok=True)
    img = cv2.imread(img_path)

    result = table_engine(img)

    def custom_draw_result(image, result, font_path='simfang.ttf'):
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype(font_path, 20)
            bbox = font.getbbox("Test")
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            font = ImageFont.load_default()
            bbox = font.getbbox("Test")
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        for region in result:
            # 转换numpy数组为普通list
            bbox = [int(x) for x in region['bbox']]
            draw.rectangle(bbox, outline=(0, 255, 0), width=2)
            draw.text((bbox[0], bbox[1] - text_h),
                      region['type'], fill=(255, 0, 0), font=font)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    vis_img = custom_draw_result(img, result)
    cv2.imwrite('./output/layout_vis.jpg', vis_img)

    #转换numpy数据
    serializable_result = []
    for region in result:
        processed = {
            'type': region['type'],
            'bbox': [float(x) for x in region['bbox']],  # 显式转换numpy.float32
            'confidence': float(region.get('confidence', 0))
        }
        if 'text' in region:
            processed['text'] = region['text']
        serializable_result.append(processed)

    # 保存JSON
    with open('./output/res.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f,
                  ensure_ascii=False,
                  indent=2,
                  cls=NumpyEncoder)

    # 打印结果
    for region in serializable_result:
        print(f"类型: {region['type']}")
        print(f"坐标: {region['bbox']}")
        if 'text' in region:
            print(f"内容: {region['text']}")
        print()

analyze_layout("layout.jpg")

"""

from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayout_plus-L")
output = model.predict("layout.jpg", batch_size=1, layout_nms=True)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
