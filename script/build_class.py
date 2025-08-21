#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
遍历目录下的img 和 label文件 更改class
@Project ：inference_recognition 
@File    ：build_class.py
@Author  ：早日赚到一个易同学
@Date    ：2025/4/8 17:33 
'''



import os

def process_yolo_labels(folder):
    txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
    jpg_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]

    print(f"共找到 {len(txt_files)} 个 label 文件，{len(jpg_files)} 个图片文件")

    # 第一步：修改标签为 2，删除空的txt对应的jpg
    for txt_file in txt_files:
        txt_path = os.path.join(folder, txt_file)
        jpg_path = os.path.join(folder, txt_file.replace('.txt', '.jpg'))

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            # 标签为空，删除图片和txt
            print(f"[空文件] 删除：{txt_path} 和 {jpg_path}")
            os.remove(txt_path)
            if os.path.exists(jpg_path):
                os.remove(jpg_path)
        else:
            # 修改 class 为 2
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    parts[0] = '1'  # 修改 class_id
                    new_lines.append(' '.join(parts))
            # 重新写入文件
            with open(txt_path, 'w') as f:
                f.write('\n'.join(new_lines) + '\n')

    # 第二步：检查多余的 jpg 是否有对应 txt，没有则删除 jpg
    current_txt_set = {f.replace('.txt', '') for f in os.listdir(folder) if f.endswith('.txt')}
    for jpg_file in jpg_files:
        base_name = jpg_file.replace('.jpg', '')
        if base_name not in current_txt_set:
            jpg_path = os.path.join(folder, jpg_file)
            print(f"[无标签] 删除多余图片：{jpg_path}")
            os.remove(jpg_path)

    print("处理完成")

# 用法
if __name__ == "__main__":
    your_folder = "C:\资料\选择题数据集 线上正常（完成校对\校对"  # 👈 修改为你的目录
    process_yolo_labels(your_folder)
