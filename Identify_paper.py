import os
import sys
import glob
import string
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import pandas as pd

from PyQt5 import QtCore, QtGui, QtWidgets
from torchvision.ops import nms


# =============================
# 数据结构
# =============================
@dataclass
class RecognizeConfig:
    rows: int = 1
    cols: int = 1
    options_per_question: int = 4  # 每题选项数，默认ABCD
    selection_norm: Optional[QtCore.QRectF] = None  # 归一化坐标 [0,1]
    yolo_model_path: Optional[str] = None  # YOLO 模型路径（识别每个选项区域）
    show_preview: bool = True  # 是否在界面上覆盖显示识别结果


# =============================
# 图形视图：支持橡皮筋框选 + 显示覆盖图
# =============================
class ImageView(QtWidgets.QGraphicsView):
    selectionChanged = QtCore.pyqtSignal(QtCore.QRectF)  # 图像坐标（像素）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = None
        self._rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._origin = QtCore.QPoint()
        self._selecting = False

        self._image = None  # numpy BGR
        self._image_qpix: Optional[QtGui.QPixmap] = None

    def load_image(self, image_path: str):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法加载图片: {image_path}")
        self.show_image_array(img)

    def show_image_array(self, img: np.ndarray):
        """直接以 numpy BGR 图像显示。"""
        self._image = img
        h, w = img.shape[:2]
        qimg = QtGui.QImage(img.data, w, h, img.strides[0], QtGui.QImage.Format_BGR888)
        self._image_qpix = QtGui.QPixmap.fromImage(qimg)

        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(self._image_qpix)
        self.setSceneRect(0, 0, w, h)
        self.fitInView(self._pixmap_item, QtCore.Qt.KeepAspectRatio)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, QtCore.Qt.KeepAspectRatio)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._image_qpix is not None:
            self._origin = event.pos()
            self._rubber_band.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._rubber_band.show()
            self._selecting = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._selecting:
            rect = QtCore.QRect(self._origin, event.pos()).normalized()
            self._rubber_band.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._selecting:
            self._selecting = False
            self._rubber_band.hide()
            # 将视口坐标转换为场景坐标，再到图像像素坐标
            vistart = self.mapToScene(self._rubber_band.geometry().topLeft())
            viend = self.mapToScene(self._rubber_band.geometry().bottomRight())
            img_rect = QtCore.QRectF(vistart, viend)
            img_rect = img_rect.intersected(self.sceneRect())
            if img_rect.isValid() and img_rect.width() > 5 and img_rect.height() > 5:
                self.selectionChanged.emit(img_rect)
        super().mouseReleaseEvent(event)

    def image_size(self) -> Tuple[int, int]:
        if self._image is None:
            return (0, 0)
        h, w = self._image.shape[:2]
        return (w, h)


# =============================
# 识别核心逻辑
# =============================
LETTERS = list(string.ascii_uppercase)


def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def detect_choice_answer(image, option):
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 二值化（阈值可以根据试卷深浅调整）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 此时黑色（涂黑的地方）是白色像素(255)，背景是0

    h, w = binary.shape
    block_width = w // option

    counts = []
    for i in range(option):
        # 每一块区域
        x_start = i * block_width
        x_end = (i + 1) * block_width if i < option - 1 else w
        block = binary[:, x_start:x_end]

        # 统计非零像素数量（即黑色填涂的面积）
        black_count = cv2.countNonZero(block)
        if black_count < 150 :
            black_count = 0
        counts.append(black_count)
    return counts


def detect_choice_in_cell(cell_img: np.ndarray) -> float:
    """计算涂黑程度（均值越大越黑）。"""
    g = to_gray(cell_img)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    inv = 255 - g.astype(np.float32)
    return float(np.mean(inv))


def rect_from_norm(norm: QtCore.QRectF, img_w: int, img_h: int) -> QtCore.QRect:
    return QtCore.QRect(
        int(norm.x() * img_w),
        int(norm.y() * img_h),
        int(norm.width() * img_w),
        int(norm.height() * img_h),
    )


def norm_from_rect(rect: QtCore.QRectF, img_w: int, img_h: int) -> QtCore.QRectF:
    return QtCore.QRectF(
        rect.x() / img_w, rect.y() / img_h, rect.width() / img_w, rect.height() / img_h
    )


# =============================
# 主窗口
# =============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._yolo_model = None  # YOLO 模型实例
        self.setWindowTitle("试卷选择题识别 · YOLO选项区域 + 预览")
        self.resize(1280, 840)

        # 状态
        self.cfg = RecognizeConfig()
        self.image_paths: List[str] = []
        self.current_index = 0

        # UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 3)
        layout.addLayout(right, 2)

        # 图像视图
        self.view = ImageView()
        self.view.selectionChanged.connect(self.on_selection_changed)
        left.addWidget(self.view, stretch=1)

        # 导航条
        nav = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("上一张")
        self.btn_next = QtWidgets.QPushButton("下一张")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        left.addLayout(nav)

        # —— 右侧控制面板 ——
        form = QtWidgets.QFormLayout()

        # 目录选择
        self.le_folder = QtWidgets.QLineEdit()
        self.btn_browse = QtWidgets.QPushButton("选择目录…")
        self.btn_browse.clicked.connect(self.choose_folder)
        h_folder = QtWidgets.QHBoxLayout()
        h_folder.addWidget(self.le_folder)
        h_folder.addWidget(self.btn_browse)
        w_folder = QtWidgets.QWidget()
        w_folder.setLayout(h_folder)
        form.addRow("图片目录:", w_folder)

        # YOLO（必填）
        self.le_yolo = QtWidgets.QLineEdit()
        self.le_yolo.setPlaceholderText("YOLO 模型路径 .pt/.onnx")
        form.addRow("YOLO模型路径:", self.le_yolo)

        # 行列数
        self.spin_rows = QtWidgets.QSpinBox()
        self.spin_rows.setRange(1, 100)
        self.spin_rows.setValue(3)
        self.spin_rows.valueChanged.connect(self.on_grid_changed)
        form.addRow("题目行数:", self.spin_rows)

        self.spin_cols = QtWidgets.QSpinBox()
        self.spin_cols.setRange(1, 100)
        self.spin_cols.setValue(4)
        self.spin_cols.valueChanged.connect(self.on_grid_changed)
        form.addRow("每行题数:", self.spin_cols)

        # 每题选项数（A-D/E…)
        self.spin_opts = QtWidgets.QSpinBox()
        self.spin_opts.setRange(2, 10)
        self.spin_opts.setValue(4)
        self.spin_opts.valueChanged.connect(self.on_options_changed)
        form.addRow("每题选项数:", self.spin_opts)

        # 预览开关 + 预览当前按钮
        self.cb_preview = QtWidgets.QCheckBox("显示识别结果预览")
        self.cb_preview.setChecked(True)
        self.btn_preview_current = QtWidgets.QPushButton("预览当前图")
        self.btn_preview_current.clicked.connect(self.preview_current_image)
        preview_row = QtWidgets.QHBoxLayout()
        preview_row.addWidget(self.cb_preview)
        preview_row.addWidget(self.btn_preview_current)
        preview_w = QtWidgets.QWidget()
        preview_w.setLayout(preview_row)
        form.addRow("预览:", preview_w)

        # 答案表
        self.table_answers = QtWidgets.QTableWidget(3, 4)
        self.table_answers.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_answers.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.fill_table_with_default()
        form.addRow("标准答案(行×列):", self.table_answers)

        # 操作按钮
        self.btn_export = QtWidgets.QPushButton("识别全部并导出 Excel")
        self.btn_export.clicked.connect(self.recognize_all_and_export)
        form.addRow(self.btn_export)

        # 日志
        self.te_log = QtWidgets.QPlainTextEdit()
        self.te_log.setReadOnly(True)
        form.addRow("日志:", self.te_log)

        right.addLayout(form)

        self.set_controls_enabled(False)

    # ============ UI 辅助 ============
    def set_controls_enabled(self, enabled: bool):
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
        self.btn_export.setEnabled(enabled)
        self.spin_rows.setEnabled(enabled)
        self.spin_cols.setEnabled(enabled)
        self.spin_opts.setEnabled(enabled)
        self.table_answers.setEnabled(enabled)
        self.cb_preview.setEnabled(enabled)
        self.btn_preview_current.setEnabled(enabled)

    def log(self, msg: str):
        self.te_log.appendPlainText(msg)
        self.te_log.verticalScrollBar().setValue(self.te_log.verticalScrollBar().maximum())

    def choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择包含试卷图片的目录")
        if folder:
            self.le_folder.setText(folder)
            self.load_folder(folder)

    def load_folder(self, folder: str):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        paths = []
        for ext in exts:
            paths += glob.glob(os.path.join(folder, ext))
        paths.sort()

        if not paths:
            QtWidgets.QMessageBox.warning(self, "提示", "该目录下未找到图片文件")
            self.set_controls_enabled(False)
            return

        self.image_paths = paths
        self.current_index = 0
        self.show_current_image()
        self.set_controls_enabled(True)
        self.log(f"加载 {len(paths)} 张图片。可在左侧图像上拖拽框选整块题区。")

    def show_current_image(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.view.show_image_array(img)
        self.setWindowTitle(f"试卷选择题识别 · {os.path.basename(path)} ({self.current_index+1}/{len(self.image_paths)})")

    def prev_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.show_current_image()

    def next_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.show_current_image()

    # ============ 选区 & 表格 ============
    def on_selection_changed(self, rect_img_coords: QtCore.QRectF):
        w, h = self.view.image_size()
        if w == 0 or h == 0:
            return
        self.cfg.selection_norm = norm_from_rect(rect_img_coords, w, h)
        x, y, rw, rh = int(rect_img_coords.x()), int(rect_img_coords.y()), int(rect_img_coords.width()), int(rect_img_coords.height())
        self.log(f"选区更新: x={x}, y={y}, w={rw}, h={rh} (将对YOLO进行裁剪检测)")

    def on_grid_changed(self):
        rows = self.spin_rows.value()
        cols = self.spin_cols.value()
        self.table_answers.setRowCount(rows)
        self.table_answers.setColumnCount(cols)
        # 每次刷新默认填A
        for r in range(rows):
            for c in range(cols):
                self._set_table_cell_combo(r, c, default_letter="A")

    def on_options_changed(self):
        # 更新表格中所有下拉选项的候选范围（A..）
        rows = self.table_answers.rowCount()
        cols = self.table_answers.columnCount()
        for r in range(rows):
            for c in range(cols):
                combo = QtWidgets.QComboBox()
                combo.addItems(LETTERS[: self.spin_opts.value()])
                self.table_answers.setCellWidget(r, c, combo)

    def fill_table_with_default(self):
        rows = self.table_answers.rowCount()
        cols = self.table_answers.columnCount()
        for r in range(rows):
            for c in range(cols):
                self._set_table_cell_combo(r, c, default_letter="A")

    def _set_table_cell_combo(self, r: int, c: int, default_letter: str = "A"):
        combo = QtWidgets.QComboBox()
        k = self.spin_opts.value() if hasattr(self, 'spin_opts') else 4
        combo.addItems(LETTERS[:k])
        idx = combo.findText(default_letter)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.table_answers.setCellWidget(r, c, combo)

    def read_standard_answers(self) -> List[List[str]]:
        rows = self.table_answers.rowCount()
        cols = self.table_answers.columnCount()
        std = []
        for r in range(rows):
            row_ans = []
            for c in range(cols):
                w = self.table_answers.cellWidget(r, c)
                letter = w.currentText() if isinstance(w, QtWidgets.QComboBox) else "A"
                row_ans.append(letter)
            std.append(row_ans)
        return std

    # ============ 预览当前图 ============
    def preview_current_image(self):
        if not self.cb_preview.isChecked():
            QtWidgets.QMessageBox.information(self, "提示", "请勾选‘显示识别结果预览’后再试。")
            return
        if not self.image_paths:
            return
        # 临时对当前图片做一次检测并覆盖显示

        if not self.le_yolo.text():
            QtWidgets.QMessageBox.warning(self, "提示", "请提供YOLO模型路径（识别选项区域）。")
            return
        try:
            if not hasattr(self, "_yolo_model") or self._yolo_model is None:
                from ultralytics import YOLOv10
                if self.le_yolo.text():
                    self._yolo_model = YOLOv10(self.le_yolo.text())
                else:
                    model_path = self.resource_path("runs/detect/train10/weights/best.pt")
                    self._yolo_model = YOLOv10(model_path)
        except Exception as e:
            self.log(f"  - YOLO预测失败: {e}")
            self.log(traceback.format_exc())
            return None

        self._recognize_single(self.current_index, export_only=False)

    # ============ 识别与导出 ============
    def recognize_all_and_export(self):
        if not self.image_paths:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择图片目录")
            return

        self.cfg.rows = self.spin_rows.value()
        self.cfg.cols = self.spin_cols.value()
        self.cfg.options_per_question = self.spin_opts.value()
        self.cfg.yolo_model_path = self.le_yolo.text().strip() or None
        self.cfg.show_preview = self.cb_preview.isChecked()

        if not self.le_yolo.text():
            QtWidgets.QMessageBox.warning(self, "提示", "请提供YOLO模型路径（识别选项区域）。")
            return
        if self.cfg.selection_norm is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在左侧图像上拖拽框选整块题区。")
            return

        std_answers = self.read_standard_answers()

        # 加载 YOLO
        try:
            from ultralytics import YOLOv10
            if self.le_yolo.text():
                # self._yolo_model = YOLO(self.cfg.yolo_model_path)
                self._yolo_model = YOLOv10(self.le_yolo.text())
            else:
                model_path = self.resource_path("runs/detect/train10/weights/best.pt")
                self._yolo_model = YOLOv10(model_path)
                # self._yolo_model = YOLOv10('best.pt')
            self.log("已加载 YOLO 模型。")
        except Exception as e:
            self.log(f"[错误] 加载YOLO失败: {e}")
            return

        records = []
        for idx in range(len(self.image_paths)):
            row_dict = self._recognize_single(idx, export_only=True, std_answers=std_answers)
            if row_dict is not None:
                records.append(row_dict)

        if not records:
            QtWidgets.QMessageBox.information(self, "结果", "没有可导出的识别结果。")
            return

        df = pd.DataFrame(records)
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存结果为 Excel", "识别结果.xlsx", "Excel 文件 (*.xlsx)")
        if not save_path:
            return
        try:
            df.to_excel(save_path, index=False)
            self.log(f"已导出: {save_path}")
            QtWidgets.QMessageBox.information(self, "完成", f"导出成功:\n{save_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"导出失败: {e}")

    # ============ 核心：识别单张（可预览） ============
    def _recognize_single(self, idx: int, export_only: bool, std_answers: Optional[List[List[str]]] = None) -> Optional[Dict[str, object]]:
        path = self.image_paths[idx]
        self.log(f"处理 {idx+1}/{len(self.image_paths)}: {os.path.basename(path)}")
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.log("  - 无法读取，跳过")
            return None

        h, w = img.shape[:2]
        if self.cfg.selection_norm is None:
            self.log("  - 未设置题区选区，跳过")
            return None
        rect = rect_from_norm(self.cfg.selection_norm, w, h)
        x0, y0, rw, rh = rect.x(), rect.y(), rect.width(), rect.height()
        x1, y1 = x0 + rw, y0 + rh
        # 裁剪题区
        region = img[y0:y1, x0:x1]

        res = self._yolo_model.predict(region, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu()
        scores = res.boxes.conf.cpu()           # 置信度
        keep = nms(xyxy, scores, iou_threshold=0.5)
        question_map = []
        grid = self.split_image_to_grid(region)
        for idx in keep:
            keep_x1, keep_y1, keep_x2, keep_y2 = map(int, xyxy[idx])
            # w = x2 - x1
            # h = y2 - y1
            opt_img = region[keep_y1:keep_y2, keep_x1:keep_x2]
            counts = detect_choice_answer(opt_img, self.cfg.options_per_question)
            ans_list = [i for i, v in enumerate(counts) if v != 0]
            opt_letter = "".join(LETTERS[i] for i in ans_list)
            question_map.append({
                'x1': keep_x1, 'y1': keep_y1, 'x2': keep_x2, 'y2': keep_y2,
                'opt_letter': opt_letter,
                'score': detect_choice_in_cell(opt_img),
            })
        # 排序question_map
        results_dict = self.assign_to_grid(question_map, region)

        preview = img.copy()
        # 绘制矩形边框（可选）
        cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 255, 0), 2)
        score_list = []
        for r, row in enumerate(grid):
            tmp_ans = []
            for c, (c_x, c_y, c_w, c_h) in enumerate(row):
                key = f"{r}_{c}"
                text = results_dict.get(key, "?")  if  results_dict.get(key, "?") else "?"# 没找到就用问号
                # 取格子中心作为绘制位置
                cx = int(c_x + c_w / 2)
                cy = int(c_y + c_h / 2)
                # score_dict.update({key: text})
                tmp_ans.append(text)

                # 绘制选项边框（可选）
                cv2.rectangle(preview, (x0 + c_x, y0 + c_y), (x0 + c_x+c_w, y0 + c_y + c_h), (255, 0, 0), 2)

                # 绘制文字
                cv2.putText(preview, text, (cx - 10+x0, cy + 5+y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2, cv2.LINE_AA)
            score_list.append(tmp_ans)


        self.view.show_image_array(preview)
        excel_data = {"filename": os.path.basename(path)}
        if std_answers:
            for i in range(len(std_answers)):
                for j, value in enumerate(std_answers[i]):
                    answers_key = f"{i}_{j}_answers"
                    detect_key = f"{i}_{j}_detect"
                    excel_data[answers_key] = std_answers[i][j]
                    excel_data[detect_key] = score_list[i][j]

        if export_only is False and std_answers is None:
            return None


        return excel_data

    def resource_path(self, relative_path):
        """PyInstaller 打包后资源路径"""
        if getattr(sys, 'frozen', False):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)


    def split_image_to_grid(self, image):
        """
        把图片均匀切成 rows x cols 的矩阵
        返回: [[(x,y,w,h), ...], ...]  # 二维list
        """
        rows =self.spin_rows.value()
        cols =self.spin_cols.value()
        h, w = image.shape[:2]
        cell_h = h // rows
        cell_w = w // cols

        grid = []
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                x = c * cell_w
                y = r * cell_h
                # 最后一行/列用剩余避免除不尽
                w_cell = cell_w if c < cols - 1 else w - x
                h_cell = cell_h if r < rows - 1 else h - y
                row_cells.append((x, y, w_cell, h_cell))
            grid.append(row_cells)

        return grid

    def assign_to_grid(self, detections, image):
        """
        把YOLO检测结果映射到rows x cols的矩阵格子
        detections: [{'x1':..., 'y1':..., 'x2':..., 'y2':..., 'opt_letter':..., 'score':...}]
        image_w, image_h: 图片宽高
        返回: [(row, col, opt_letter), ...]
        """
        rows = self.spin_rows.value()
        cols = self.spin_cols.value()
        h, w = image.shape[:2]
        cell_h = h // rows
        cell_w = w // cols


        results = []
        results_dict = {}
        for det in detections:
            # 框中心点
            cx = (det['x1'] + det['x2']) / 2
            cy = (det['y1'] + det['y2']) / 2

            # 落在哪个格子
            col = int(cx // cell_w)
            row = int(cy // cell_h)

            # 边界保护
            col = min(max(col, 0), cols - 1)
            row = min(max(row, 0), rows - 1)

            results.append((row, col, det['opt_letter']))
            results_dict.update({"{}_{}".format(row, col): det['opt_letter']})

        return results_dict


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys

    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: None
    cv2.destroyAllWindows = lambda *args, **kwargs: None

    # 修复 sys.stdout 在 --noconsole 下为 None 的问题
    if sys.stdout is None:
        import io

        sys.stdout = io.StringIO()
    if sys.stderr is None:
        import io

        sys.stderr = io.StringIO()
    main()
