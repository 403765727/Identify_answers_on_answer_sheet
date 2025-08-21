# import cv2
# from ultralytics import YOLOv10
#
# # 加载模型
# # model = YOLOv10("yolov10m.pt")
# model = YOLOv10(r"G:/yolov10/runs/detect/train/weights/best.pt")
#
# # 批量运算
# results = model([r"G:\yolov10\datas\train\images\20241028113706-VjpkEd.jpg"], stream=True)
#
# for result in results:
#     boxes_cls_len = len(result.boxes.cls)
#     if not boxes_cls_len:
#         # 没有检测到内容
#         continue
#     for boxes_cls_index in range(boxes_cls_len):
#         # 获取类别id
#         class_id = int(result.boxes.cls[boxes_cls_index].item())
#         # 获取类别名称
#         class_name = result.names[class_id]
#
#         # 获取相似度
#         similarity = result.boxes.conf[boxes_cls_index].item()
#
#         # 获取坐标值，左上角 和 右下角：lt_rb的值：[1145.1351318359375, 432.6763000488281, 1275.398681640625, 749.5224609375]
#         lt_rb = result.boxes.xyxy[boxes_cls_index].tolist()
#         # 转为：[[1145.1351318359375, 432.6763000488281], [1275.398681640625, 749.5224609375]]
#         lt_rb = [[lt_rb[0], lt_rb[1]], [lt_rb[0], lt_rb[1]]]
#
#         print("类别：", class_name, "相似度：", similarity, "坐标：", lt_rb)
#
#     # 图片展示
#     annotated_image = result.plot()
#     annotated_image = annotated_image[:, :, ::-1]
#     if annotated_image is not None:
#         cv2.imshow("Annotated Image", annotated_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()




from ultralytics import YOLO

# Load a pre-trained YOLOv10n model
model = YOLO(r"G:/yolov10/runs/detect/train/weights/best.pt")

# Perform object detection on an image
results = model(r"G:\yolov10\datas\train\images\20241028113706-VjpkEd.jpg")

# Display the results
results[0].show()