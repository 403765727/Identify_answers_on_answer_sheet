#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：yolov10 
@File    ：train.py
@Author  ：yiiich
@Date    ：2024/11/22 10:30 
'''
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLOv10

if __name__ == '__main__':
    model = YOLOv10('yolov10n.yaml')  # 指定YOLO模型对象，并加载指定配置文件中的模型配置
    # model.load('yolov10s.pt')      #加载预训练的权重文件'yolov10s.pt'，加速训练并提升模型性能
    # model = YOLOv10('yolov10n.yaml').load('yolov10n.pt')  # 迁移学习
    model.train(data='G:\yolov10\get_png\choice_label_data2\data.yaml',  # 指定训练数据集的配置文件路径，这个.yaml文件包含了数据集的路径和类别信息
                cache=False,  # 是否缓存数据集以加快后续训练速度，False表示不缓存
                imgsz=1088,  # 指定训练时使用的图像尺寸，640表示将输入图像调整为640x640像素
                epochs=200,  # 设置训练的总轮数为200轮
                batch=16,  # 设置每个训练批次的大小为16，即每次更新模型时使用16张图片v
                close_mosaic=10,  # 设置在训练结束前多少轮关闭 Mosaic 数据增强，10 表示在训练的最后 10 轮中关闭 Mosaic
                workers=24, # 设置用于数据加载的线程数为8，更多线程可以加快数据加载速度
                patience=50,  # 在训练时，如果经过50轮性能没有提升，则停止训练（早停机制）
                device='0',  # 指定使用的设备，'0'表示使用第一块GPU进行训练
                optimizer='SGD',  # 设置优化器为SGD（随机梯度下降），用于模型参数更新
                )


# from ultralytics import YOLOv10
# if __name__ == '__main__':
# # Load a model # 三选一
#     model = YOLOv10('yolov10n.yaml')  # build a new model from YAML
#     # model = YOLOv10('yolov10n.pt')  # load a pretrained model (recommended for training)
#     # model = YOLOv10('yolov10n.yaml').load('yolov10n.pt')  # build from YAML and transfer weights
#
#     # Train the model
#     model.train(data='G:\yolov10\datas\data.yaml', epochs=100, imgsz=640)
