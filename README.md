## 答题卡选择题自动识别说明

```txt
本项目依赖yolo10，以使用少量数据集训练出选择题目标检测模型，达到自动识别答题卡选择题答案目的。

```



#### 模型位置：

```txt
.\runs\detect\train10\weights\best.pt
```



#### 测试图片数据位置：

```txt
.\test_images
```



#### 打包体验使用方式：

```txt
1.先在目录中打开exe执行文件, 犹豫需要加载机器学习环境，需要一点等待时间。
.\dist\identify_paper.exe
```

![1](.\dist\1.png)

```txt
2. 选择答题卡图片目录.\test_images
```

![image-20250820173937524](.\dist\2.png)

```txt
3.输入yolo模型地址.\runs\detect\train10\weights\best.pt
```

![image-20250820174245738](.\dist\3.png)

```txt
4.设置行列和选项个数，框选选择题区域
```

![image-20250820174417169](.\dist\4.png)

```txt
5.预览查看结果
```

![image-20250820174527776](.\dist\5.png)

```txt
6.导出全部结果
```

![image-20250820174650455](.\dist\6.png)
