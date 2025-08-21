# -*- coding: utf-8 -*-
import cv2

from ultralytics import YOLOv10

# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt


# model = YOLOv10(r'D:\2-Python\1-YOLO\YOLOv10\yolov10-main\yolov10n.pt')
# model = YOLOv10(r'G:/yolov10/runs/detect/train2/weights/best.pt')
model = YOLOv10(r'./runs/detect/train10/weights/best.pt')


image_path = r'./tmp/1.png'
# results = model.predict(source=image_path, save=True, line_thickness=1)
results = model.predict(source=image_path, line_thickness=1)
yolo_coords = results[0].boxes.xywhn.cpu().numpy()

# 读取图片
 # 替换为你的图片路径
image = cv2.imread(image_path)

# 获取图片的宽高
height, width, _ = image.shape
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
    converted_coords.append([x1, y1, new_w, new_h])
    #
    # # 保存转换后的 YOLO 格式坐标
    # converted_coords.append([x1, y1, new_w, new_h])

    cv2.rectangle(image, (x1, y1), (x1 + new_w, y1 + new_h), (0, 255, 0), 2)
    # 绘制矩形框
    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 显示图片
cv2.imshow('Image with YOLO Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(converted_coords)