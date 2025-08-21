#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：yolov10 
@File    ：verify.py
@Author  ：yiiich
@Date    ：2024/11/22 13:44 
'''
from ultralytics import YOLOv10


def main():
    # 加载模型，split='test'利用测试集进行测试
    model = YOLOv10(r"G:/yolov10/runs/detect/train4/weights/best.pt")
    model.val(data='G:/yolov10/datas/data.yaml', split='test', imgsz=640, batch=16, device=0, workers=8)  # 模型验证


if __name__ == "__main__":
    main()
