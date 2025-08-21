import json

import cv2
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
# import torch
from PIL import Image
import io
import numpy as np

from ultralytics import YOLOv10

# 加载YOLO模型
model = YOLOv10(r'./runs/detect/train10/weights/best.pt')
app = FastAPI()


@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()  # 读取上传的图像数据

        # 将字节数据转换为 NumPy 数组
        nparr = np.frombuffer(image_data, np.uint8)

        # 使用 OpenCV 解码图像
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 获取图像的宽度和高度
        height, width, _ = image.shape
        results = model.predict(source=image, line_thickness=1)
        yolo_coords = results[0].boxes.xywhn.cpu().numpy()
        print(height, width)
        converted_coords = []
        # 转换为左上角坐标并绘制框
        for coord in yolo_coords:
            x_center, y_center, w, h = coord

            # 转换为左上角坐标
            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)

            # x1 = int((x_center - w / 2) * width)
            # y1 = int((y_center - h / 2) * height)

            # 计算转换后的宽度和高度
            new_w = int(w * width)
            new_h = int(h * height)

            # 将转换后的坐标保存
            # converted_coords.append([x1, y1, new_w, new_h])
            converted_coords.append({
                "x": x1,
                "y": y1,
                "w": new_w,
                "h":new_h})
            #
            # # 保存转换后的 YOLO 格式坐标
            # converted_coords.append([x1, y1, new_w, new_h])

            # cv2.rectangle(image, (x1, y1), (x1 + new_w, y1 + new_h), (0, 255, 0), 2)
            # 绘制矩形框
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 显示图片
        # cv2.imshow('Image with YOLO Boxes', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(converted_coords)


        # return json.dumps(converted_coords)
        return converted_coords
    except Exception as e:
        return {"error": str(e)}





# 启动命令uvicorn app:app --host 0.0.0.0 --port 8000 --reload
