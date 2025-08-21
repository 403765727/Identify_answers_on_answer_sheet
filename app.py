import json

import cv2
from fastapi import FastAPI, File, UploadFile
from torchvision.ops import nms
from pydantic import BaseModel
# import torch
from PIL import Image
import io
import numpy as np
import torch
from ultralytics import YOLOv10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 加载YOLO模型（假设是YOLOv5）
# model = YOLOv10(r'G:/yolov10/runs/detect/train14/weights/best.pt')
# model = YOLOv10(r'G:/yolov10/runs/detect/train3/weights/best.pt')#最初版选择题识别
# model = YOLOv10(r'G:/yolov10/runs/detect/train8/weights/best.pt')#abcd题号和选项识别
# model = YOLOv10(r'G:/yolov10/runs/detect/train9/weights/best.pt').to(device)#2502版选择题识别，加入更多样本，分类需要重新处理
# model = YOLOv10(r'G:/yolov10/runs/detect/train2/weights/best.pt').to(device)#2526版本，增加了数据集
model = YOLOv10(r'./runs/detect/train10/weights/best.pt').to(device)#增加了数据集

model_all = YOLOv10(r'./runs/detect/train15/weights/best.pt').to(device)#识别点位 学号  选项

model_choice = YOLOv10(r'./runs/detect/train17/weights/best.pt').to(device)#识别选项


app = FastAPI()

@app.post("/detect_all/")
async def detect_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width, _ = image.shape

        results = model_all.predict(source=image, line_width=1)
        boxes = results[0].boxes

        xyxy = boxes.xyxy.cpu()             # (x1, y1, x2, y2)
        scores = boxes.conf.cpu()           # 置信度
        classes = boxes.cls.cpu().numpy().astype(int)  # 类别索引

        # 非极大值抑制
        keep = nms(xyxy, scores, iou_threshold=0.5)

        raw_class_names = model_all.names

        def clean_class_name(raw):
            if isinstance(raw, str):
                return raw.split(":")[-1].strip('"')
            return str(raw)

        converted_coords = []
        for idx in keep:
            x1, y1, x2, y2 = map(int, xyxy[idx])
            w = x2 - x1
            h = y2 - y1
            class_id = classes[idx]
            class_raw = raw_class_names.get(class_id, str(class_id))
            class_name = clean_class_name(class_raw)

            converted_coords.append({
                "x": x1,
                "y": y1,
                "w": w,
                "h": h,
                "class_id": int(class_id),
                "class_name": class_name
            })

        return converted_coords

    except Exception as e:
        return {"error": str(e)}



@app.post("/choice/")
async def detect_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width, _ = image.shape

        results = model_choice.predict(source=image, line_width=1)
        boxes = results[0].boxes

        xyxy = boxes.xyxy.cpu()             # (x1, y1, x2, y2)
        scores = boxes.conf.cpu()           # 置信度
        classes = boxes.cls.cpu().numpy().astype(int)  # 类别索引

        # 非极大值抑制
        keep = nms(xyxy, scores, iou_threshold=0.5)

        raw_class_names = model_choice.names

        def clean_class_name(raw):
            if isinstance(raw, str):
                return raw.split(":")[-1].strip('"')
            return str(raw)

        converted_coords = []
        for idx in keep:
            x1, y1, x2, y2 = map(int, xyxy[idx])
            w = x2 - x1
            h = y2 - y1
            class_id = classes[idx]
            class_raw = raw_class_names.get(class_id, str(class_id))
            class_name = clean_class_name(class_raw)

            converted_coords.append({
                "x": x1,
                "y": y1,
                "w": w,
                "h": h,
                "class_id": int(class_id),
                "class_name": class_name
            })

        return converted_coords

    except Exception as e:
        return {"error": str(e)}



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
        # print(height, width)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)


# 启动命令uvicorn app:app --host 0.0.0.0 --port 8000 --reload