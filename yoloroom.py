# Copyright (c) 2026 Jifeng Wu
# Licensed under the MIT License. See LICENSE file in the project root for full license information
import argparse
import os
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
from pyqtcompat import (
    QApplication,
    QCheckBox,
    QColor,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QImage,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPainter,
    QPen,
    QPixmap,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QtSignal,
    QSIZEPOLICY_EXPANDING,
    QT_ALIGN_CENTER,
    QT_KEEP_ASPECT_RATIO,
    QT_SCROLLBAR_ALWAYS_OFF,
    QT_SCROLLBAR_AS_NEEDED,
    QT_SMOOTH_TRANSFORMATION,
    QVBoxLayout,
    QWidget,
    exec_qapplication,
    Format_RGB888,
)
from ultralytics import YOLO

WIDTH = 1280
HEIGHT = 720
THUMBNAIL_SIZE = 128
SIDEBAR_WIDTH = 480
DETECTION_WIDTH = 4
DEFAULT_PRETRAINED_ULTRALYTICS_YOLO_MODEL = 'yolo26n.pt'
CLASS_COLOR_CACHE = {}  # type: Dict[str, QColor]



def get_class_color(class_name):
    # type: (str) -> QColor
    if class_name not in CLASS_COLOR_CACHE:
        seed = 0
        index = 0
        for character in class_name:
            seed = seed + ((index + 1) * ord(character))
            index = index + 1
        red = 20 + (seed * 53) % 216
        green = 20 + (seed * 97) % 216
        blue = 20 + (seed * 193) % 216
        CLASS_COLOR_CACHE[class_name] = QColor(red, green, blue, 170)
    return CLASS_COLOR_CACHE[class_name]



def show_current_exception_message_box(parent, title):
    # type: (Any, str) -> None
    exc_info = sys.exc_info()
    exc_type = exc_info[0]
    exc_value = exc_info[1]
    if exc_type is None:
        QMessageBox.critical(parent, title, 'Unknown error.')
        return
    QMessageBox.critical(parent, title, '%s: %s' % (exc_type.__name__, exc_value))



def get_ultralytics_model_class_names(yolo_model):
    # type: (Any) -> List[str]
    model_names = yolo_model.names
    if isinstance(model_names, dict):
        class_indices = sorted(model_names.keys())
        return [model_names[class_index] for class_index in class_indices]
    return list(model_names)



def get_ultralytics_class_name_to_index(yolo_model):
    # type: (Any) -> Dict[str, int]
    model_names = yolo_model.names
    class_name_to_index = {}
    if isinstance(model_names, dict):
        for class_index in sorted(model_names.keys()):
            class_name_to_index[model_names[class_index]] = class_index
        return class_name_to_index

    class_index = 0
    for class_name in model_names:
        class_name_to_index[class_name] = class_index
        class_index = class_index + 1
    return class_name_to_index



def clamp_detection_bbox(x1, y1, x2, y2, image_width, image_height):
    # type: (float, float, float, float, int, int) -> Tuple[int, int, int, int]
    x = int(round(x1))
    y = int(round(y1))
    right = int(round(x2))
    bottom = int(round(y2))

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if right > image_width:
        right = image_width
    if bottom > image_height:
        bottom = image_height

    width = right - x
    height = bottom - y
    if width < 1:
        width = 1
    if height < 1:
        height = 1
    if x + width > image_width:
        width = image_width - x
    if y + height > image_height:
        height = image_height - y

    return (x, y, width, height)



class Detection(object):
    __slots__ = ['class_name', 'bbox', 'confidence', 'checked']

    def __init__(self, class_name, bbox, confidence):
        # type: (str, Tuple[int, int, int, int], float) -> None
        self.class_name = class_name  # type: str
        self.bbox = bbox  # type: Tuple[int, int, int, int]
        self.confidence = confidence  # type: float
        self.checked = True  # type: bool



def create_detection(class_name, bbox, confidence):
    # type: (str, Tuple[int, int, int, int], float) -> Detection
    return Detection(class_name, bbox, confidence)



def get_detection_label_text(detection):
    # type: (Detection) -> str
    return '%s %.2f' % (detection.class_name, detection.confidence)



def render_detections_on_hwc_bgr_888_ndarray(hwc_bgr_888_ndarray, detections):
    # type: (Any, List[Detection]) -> Any
    rendered = hwc_bgr_888_ndarray.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    for detection in detections:
        if not detection.checked:
            continue

        class_color = get_class_color(detection.class_name)
        bgr_color = (class_color.blue(), class_color.green(), class_color.red())
        x, y, box_width, box_height = detection.bbox
        right = x + box_width
        bottom = y + box_height
        cv2.rectangle(rendered, (x, y), (right, bottom), bgr_color, DETECTION_WIDTH)

        label_text = get_detection_label_text(detection)
        text_size, baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        text_width, text_height = text_size
        label_left = x
        label_top = y - text_height - baseline - 8
        if label_top < 0:
            label_top = y
        label_right = label_left + text_width + 8
        label_bottom = label_top + text_height + baseline + 8
        cv2.rectangle(rendered, (label_left, label_top), (label_right, label_bottom), bgr_color, -1)
        cv2.putText(
            rendered,
            label_text,
            (label_left + 4, label_bottom - baseline - 4),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return rendered



def run_ultralytics_yolo_inference(yolo_model, filenames, hwc_bgr_888_ndarrays, selected_classes):
    # type: (Any, List[str], List[Any], Optional[List[str]]) -> List[ImageDetection]
    class_name_to_index = get_ultralytics_class_name_to_index(yolo_model)
    selected_class_indices = None
    if selected_classes:
        selected_class_indices = []
        for class_name in selected_classes:
            selected_class_indices.append(class_name_to_index[class_name])

    results = yolo_model.predict(
        source=hwc_bgr_888_ndarrays,
        classes=selected_class_indices,
        verbose=False,
    )

    images = []
    image_index = 0
    for result in results:
        detections = []
        boxes = result.boxes
        if boxes is not None:
            xyxy_values = boxes.xyxy.cpu().tolist()
            class_values = boxes.cls.cpu().tolist()
            confidence_values = boxes.conf.cpu().tolist()
            detection_index = 0
            for class_value in class_values:
                xyxy_value = xyxy_values[detection_index]
                confidence_value = confidence_values[detection_index]
                image_array = hwc_bgr_888_ndarrays[image_index]
                image_height = image_array.shape[0]
                image_width = image_array.shape[1]
                bbox = clamp_detection_bbox(
                    xyxy_value[0],
                    xyxy_value[1],
                    xyxy_value[2],
                    xyxy_value[3],
                    image_width,
                    image_height,
                )
                class_name = result.names[int(class_value)]
                detections.append(create_detection(class_name, bbox, float(confidence_value)))
                detection_index = detection_index + 1

        images.append(
            ImageDetection(
                filename=filenames[image_index],
                hwc_bgr_888_ndarray=hwc_bgr_888_ndarrays[image_index],
                detections=detections,
            )
        )
        image_index = image_index + 1

    return images



def replace_name_template_variables(name_template, filename, class_name, index_text):
    # type: (str, str, str, str) -> str
    output = name_template
    output = output.replace('{filename}', filename)
    output = output.replace('{class}', class_name)
    output = output.replace('{index}', index_text)
    return output



def get_open_file_dialog_file_paths(dialog_result):
    # type: (Any) -> List[str]
    if isinstance(dialog_result, tuple):
        return list(dialog_result[0])
    return list(dialog_result)



def load_images_from_paths(image_file_paths):
    # type: (List[str]) -> Tuple[List[str], List[Any]]
    filenames = []
    hwc_bgr_888_ndarrays = []
    for image_file_path in image_file_paths:
        hwc_bgr_888_ndarray = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        if hwc_bgr_888_ndarray is None:
            raise RuntimeError('Could not read image file: ' + image_file_path)

        filename = os.path.splitext(os.path.basename(image_file_path))[0]
        filenames.append(filename)
        hwc_bgr_888_ndarrays.append(hwc_bgr_888_ndarray)

    return (filenames, hwc_bgr_888_ndarrays)



def clear_layout(layout):
    # type: (Any) -> None
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())


class ImageDetection(object):
    def __init__(self, filename, hwc_bgr_888_ndarray, detections):
        # type: (str, Any, List[Detection]) -> None
        self.filename = filename  # type: str
        self.hwc_bgr_888_ndarray = hwc_bgr_888_ndarray  # type: Any
        self.height = self.hwc_bgr_888_ndarray.shape[0]  # type: int
        self.width = self.hwc_bgr_888_ndarray.shape[1]  # type: int
        self.detections = detections  # type: List[Detection]

    def create_qimage_from_hwc_bgr_888_ndarray(self, hwc_bgr_888_ndarray):
        # type: (Any) -> QImage
        hwc_rgb_888_ndarray = cv2.cvtColor(hwc_bgr_888_ndarray, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            hwc_rgb_888_ndarray.data,
            self.width,
            self.height,
            hwc_rgb_888_ndarray.strides[0],
            Format_RGB888,
        )
        return qimage.copy()

    def get_qimage(self):
        # type: () -> QImage
        return self.create_qimage_from_hwc_bgr_888_ndarray(self.hwc_bgr_888_ndarray)

    def get_annotated_qimage(self):
        # type: () -> QImage
        rendered = render_detections_on_hwc_bgr_888_ndarray(self.hwc_bgr_888_ndarray, self.detections)
        return self.create_qimage_from_hwc_bgr_888_ndarray(rendered)


class ThumbnailLabel(QLabel):
    clicked = QtSignal(int)

    def __init__(self, idx, qimg, width, height, parent=None):
        # type: (int, QImage, int, int, Optional[Any]) -> None
        super(ThumbnailLabel, self).__init__(parent)
        pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(
            pixmap.scaled(
                width,
                height,
                QT_KEEP_ASPECT_RATIO,
                QT_SMOOTH_TRANSFORMATION,
            )
        )
        self.idx = idx  # type: int
        self.setFixedSize(width, height)

    def mousePressEvent(self, event):
        # type: (Any) -> None
        QLabel.mousePressEvent(self, event)
        self.clicked.emit(self.idx)


class ImageDetectionLabel(QLabel):
    def __init__(self, parent=None):
        # type: (Optional[Any]) -> None
        super(ImageDetectionLabel, self).__init__(parent=parent)
        self.imgdet = None  # type: Optional[ImageDetection]

    def show_placeholder_text(self, text):
        # type: (str) -> None
        self.imgdet = None
        self.clear()
        self.setText(text)

    def set_imgdet(self, imgdet):
        # type: (ImageDetection) -> None
        self.imgdet = imgdet
        self.setText('')
        self.render_imgdet()

    def render_imgdet(self, width=None, height=None):
        # type: (Optional[int], Optional[int]) -> None
        if self.imgdet is None:
            return

        if width is None:
            width = self.width()
        if height is None:
            height = self.height()

        qimg = self.imgdet.get_annotated_qimage()
        pixmap = QPixmap.fromImage(qimg).scaled(
            width,
            height,
            QT_KEEP_ASPECT_RATIO,
            QT_SMOOTH_TRANSFORMATION,
        )
        self.setPixmap(pixmap)

    def resizeEvent(self, event):
        # type: (Any) -> None
        QLabel.resizeEvent(self, event)
        size = event.size()
        self.render_imgdet(size.width(), size.height())


class MainWindow(QMainWindow):
    def __init__(self, yolo_model, yolo_model_path, filenames, hwc_bgr_888_ndarrays, classes):
        # type: (Any, str, List[str], List[Any], List[str]) -> None
        super(MainWindow, self).__init__()
        self.yolo_model = yolo_model  # type: Any
        self.yolo_model_path = yolo_model_path  # type: str
        self.filenames = filenames  # type: List[str]
        self.hwc_bgr_888_ndarrays = hwc_bgr_888_ndarrays  # type: List[Any]
        self.classes = classes  # type: List[str]
        self.current_index = 0  # type: int
        self.images = []  # type: List[ImageDetection]
        self.open_image_dir = os.getcwd()  # type: str

        self.resize(WIDTH, HEIGHT)
        self.setWindowTitle('YOLOroom')

        main_widget = QWidget()
        main_hbox = QHBoxLayout()
        main_widget.setLayout(main_hbox)
        self.setCentralWidget(main_widget)

        central_vbox = QVBoxLayout()
        main_hbox.addLayout(central_vbox, 1)

        self.image_label = ImageDetectionLabel(parent=self)
        self.image_label.setAlignment(QT_ALIGN_CENTER)
        self.image_label.setSizePolicy(QSIZEPOLICY_EXPANDING, QSIZEPOLICY_EXPANDING)
        central_vbox.addWidget(self.image_label, 1)

        self.thumb_scroll_area = QScrollArea()
        self.thumb_scroll_area.setWidgetResizable(False)
        self.thumb_scroll_area.setHorizontalScrollBarPolicy(QT_SCROLLBAR_AS_NEEDED)
        self.thumb_scroll_area.setVerticalScrollBarPolicy(QT_SCROLLBAR_ALWAYS_OFF)
        self.thumb_scroll_area.setFixedHeight(THUMBNAIL_SIZE + 24)
        self.thumb_widget = QWidget()
        self.thumb_layout = QHBoxLayout()
        self.thumb_layout.setContentsMargins(0, 0, 0, 0)
        self.thumb_layout.setSpacing(6)
        self.thumb_widget.setLayout(self.thumb_layout)
        self.thumb_widget.setFixedHeight(THUMBNAIL_SIZE)
        self.thumb_scroll_area.setWidget(self.thumb_widget)
        central_vbox.addWidget(self.thumb_scroll_area, 0)

        sidebar_container = QWidget()
        sidebar = QVBoxLayout()
        sidebar_container.setLayout(sidebar)
        sidebar_container.setFixedWidth(SIDEBAR_WIDTH)
        main_hbox.addWidget(sidebar_container, 0)

        self.open_images_btn = QPushButton('Open images...')
        self.open_images_btn.clicked.connect(self.open_images)
        sidebar.addWidget(self.open_images_btn)

        self.close_current_image_btn = QPushButton('Close current image')
        self.close_current_image_btn.clicked.connect(self.close_current_image)
        sidebar.addWidget(self.close_current_image_btn)

        export_box = QGroupBox('Export')
        export_vbox = QVBoxLayout()
        self.dir_line = QLineEdit(os.getcwd())
        browse_btn = QPushButton('Browse...')
        browse_btn.clicked.connect(self.browse_dir)
        dir_hbox = QHBoxLayout()
        dir_hbox.addWidget(self.dir_line)
        dir_hbox.addWidget(browse_btn)
        export_vbox.addWidget(QLabel('Directory:'))
        export_vbox.addLayout(dir_hbox)
        self.format_combo = QComboBox()
        self.format_combo.addItems(['JPG', 'PNG'])
        export_vbox.addWidget(QLabel('Format:'))
        export_vbox.addWidget(self.format_combo)
        self.name_template = QLineEdit('{filename}_{class}_{index}')
        export_vbox.addWidget(QLabel('Name template:'))
        export_vbox.addWidget(self.name_template)
        self.export_btn = QPushButton('Export checked')
        self.export_btn.clicked.connect(self.export_checked)
        export_vbox.addWidget(self.export_btn)
        export_box.setLayout(export_vbox)

        self.class_panels_box = QGroupBox('Classes and Instances')
        self.class_panels_vbox = QVBoxLayout()
        self.class_panels_box.setLayout(self.class_panels_vbox)
        sidebar.addWidget(self.class_panels_box)

        sidebar.addStretch()
        sidebar.addWidget(export_box)

        self.rerun_detections()

    def rerun_detections(self):
        # type: () -> None
        try:
            self.images = self.run_detections_for_loaded_images(self.filenames, self.hwc_bgr_888_ndarrays)
            self.refresh_view()
        except Exception:
            show_current_exception_message_box(self, 'Detection failed')

    def run_detections_for_loaded_images(self, filenames, hwc_bgr_888_ndarrays):
        # type: (List[str], List[Any]) -> List[ImageDetection]
        if not hwc_bgr_888_ndarrays:
            return []
        return run_ultralytics_yolo_inference(
            yolo_model=self.yolo_model,
            filenames=filenames,
            hwc_bgr_888_ndarrays=hwc_bgr_888_ndarrays,
            selected_classes=self.classes,
        )

    def refresh_view(self):
        # type: () -> None
        if self.current_index >= len(self.images):
            self.current_index = 0
        self.rebuild_thumbnails()
        self.update_image()
        self.update_instances_panel()

    def rebuild_thumbnails(self):
        # type: () -> None
        clear_layout(self.thumb_layout)
        index = 0
        for imgdet in self.images:
            qimg = imgdet.get_qimage()
            thumb = ThumbnailLabel(index, qimg, THUMBNAIL_SIZE, THUMBNAIL_SIZE, parent=self)
            thumb.clicked.connect(self.select_image)
            self.thumb_layout.addWidget(thumb)
            index = index + 1

        thumbnail_count = len(self.images)
        thumbnail_spacing = self.thumb_layout.spacing()
        thumbnail_width = 0
        if thumbnail_count > 0:
            thumbnail_width = (thumbnail_count * THUMBNAIL_SIZE) + ((thumbnail_count - 1) * thumbnail_spacing)
        self.thumb_widget.setFixedSize(thumbnail_width, THUMBNAIL_SIZE)

    def select_image(self, idx):
        # type: (int) -> None
        self.current_index = idx
        self.update_image()
        self.update_instances_panel()

    def update_image(self):
        # type: () -> None
        if self.images:
            self.image_label.set_imgdet(self.images[self.current_index])
            return
        self.image_label.show_placeholder_text('No images loaded.\nClick "Open images..." to choose image files.')

    def update_instances_panel(self):
        # type: () -> None
        clear_layout(self.class_panels_vbox)

        if not self.images:
            return

        imgdet = self.images[self.current_index]
        class2detections = defaultdict(list)
        global_index = 0
        for detection in imgdet.detections:
            class2detections[detection.class_name].append((global_index, detection))
            global_index = global_index + 1

        for class_name in self.classes:
            class_detections = class2detections.get(class_name, [])
            if not class_detections:
                continue

            groupbox = QGroupBox()
            hbox = QHBoxLayout()
            class_color = get_class_color(class_name)
            color_patch = QFrame()
            color_patch.setFixedWidth(16)
            color_patch.setFixedHeight(16)
            color_patch.setStyleSheet('background: %s' % class_color.name())
            hbox.addWidget(color_patch)
            class_cb = QCheckBox(class_name)
            any_checked = any(detection.checked for (_, detection) in class_detections)
            class_cb.setChecked(any_checked)
            class_cb.stateChanged.connect(self.make_toggle_class(class_name))
            hbox.addWidget(class_cb)
            hbox.addStretch()
            groupbox.setLayout(hbox)
            self.class_panels_vbox.addWidget(groupbox)

            instance_num = 0
            for global_index, detection in class_detections:
                inst_hbox = QHBoxLayout()
                label_text = 'Instance %d (%.2f)' % (instance_num, detection.confidence)
                inst_cb = QCheckBox(label_text)
                inst_cb.setChecked(detection.checked)
                inst_cb.stateChanged.connect(self.make_instance_toggle(global_index))
                inst_hbox.addSpacing(30)
                inst_hbox.addWidget(inst_cb)
                self.class_panels_vbox.addLayout(inst_hbox)
                instance_num = instance_num + 1

    def make_instance_toggle(self, idx):
        # type: (int) -> Callable[[int], None]
        def toggle(state):
            # type: (int) -> None
            imgdet = self.images[self.current_index]
            imgdet.detections[idx].checked = bool(state)
            self.update_image()
            self.update_instances_panel()
        return toggle

    def make_toggle_class(self, class_name):
        # type: (str) -> Callable[[int], None]
        def toggle(state):
            # type: (int) -> None
            checked = bool(state)
            imgdet = self.images[self.current_index]
            for detection in imgdet.detections:
                if detection.class_name == class_name:
                    detection.checked = checked
            self.update_image()
            self.update_instances_panel()
        return toggle

    def open_images(self):
        # type: () -> None
        dialog_result = QFileDialog.getOpenFileNames(
            self,
            'Open Images',
            self.open_image_dir,
            'Images (*.bmp *.gif *.jpg *.jpeg *.png *.tif *.tiff *.webp);;All files (*)',
        )
        image_file_paths = get_open_file_dialog_file_paths(dialog_result)
        if not image_file_paths:
            return

        try:
            opened_filenames, opened_hwc_bgr_888_ndarrays = load_images_from_paths(image_file_paths)
            combined_filenames = self.filenames + opened_filenames
            combined_hwc_bgr_888_ndarrays = self.hwc_bgr_888_ndarrays + opened_hwc_bgr_888_ndarrays
            combined_images = self.run_detections_for_loaded_images(
                combined_filenames,
                combined_hwc_bgr_888_ndarrays,
            )
        except Exception:
            show_current_exception_message_box(self, 'Open images failed')
            return

        self.filenames = combined_filenames
        self.hwc_bgr_888_ndarrays = combined_hwc_bgr_888_ndarrays
        self.images = combined_images
        self.current_index = len(self.images) - len(opened_filenames)
        self.open_image_dir = os.path.dirname(image_file_paths[0])
        self.refresh_view()

    def close_current_image(self):
        # type: () -> None
        if not self.images:
            return

        del self.filenames[self.current_index]
        del self.hwc_bgr_888_ndarrays[self.current_index]
        del self.images[self.current_index]

        if self.current_index >= len(self.images) and self.current_index > 0:
            self.current_index = self.current_index - 1

        self.refresh_view()

    def browse_dir(self):
        # type: () -> None
        dirname = QFileDialog.getExistingDirectory(self, 'Select Export Directory')
        if dirname:
            self.dir_line.setText(dirname)

    def export_checked(self):
        # type: () -> None
        export_dir = self.dir_line.text()
        if not export_dir:
            QMessageBox.critical(self, 'Export failed', 'Please choose an export directory.')
            return
        if not os.path.isdir(export_dir):
            QMessageBox.critical(self, 'Export failed', 'Export directory does not exist: ' + export_dir)
            return
        if not self.images:
            QMessageBox.critical(self, 'Export failed', 'There are no opened images to export.')
            return

        file_fmt = self.format_combo.currentText().lower()
        name_template = self.name_template.text()

        try:
            exported_count = 0
            for imgdet in self.images:
                base_name = imgdet.filename
                index = 0
                for detection in imgdet.detections:
                    if detection.checked:
                        x, y, box_width, box_height = detection.bbox
                        cropped = imgdet.hwc_bgr_888_ndarray[y:y + box_height, x:x + box_width]
                        output_name = replace_name_template_variables(
                            name_template,
                            base_name,
                            detection.class_name,
                            str(index),
                        )
                        outpath = os.path.join(export_dir, output_name + '.' + file_fmt)
                        if not cv2.imwrite(outpath, cropped):
                            raise RuntimeError('Could not write file: ' + outpath)
                        exported_count = exported_count + 1
                    index = index + 1
        except Exception:
            show_current_exception_message_box(self, 'Export failed')
            return

        QMessageBox.information(self, 'Export complete', 'Exported %d checked detections.' % exported_count)



def main():
    # type: () -> None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--class',
        action='append',
        dest='classes',
        help='Class name filter. Repeat to include multiple classes. Defaults to all classes supported by the model.',
    )
    parser.add_argument(
        '--pretrained-ultralytics-yolo-model',
        type=str,
        default=DEFAULT_PRETRAINED_ULTRALYTICS_YOLO_MODEL,
        help='Pretrained Ultralytics YOLO model path or model name. Default: ' + DEFAULT_PRETRAINED_ULTRALYTICS_YOLO_MODEL,
    )
    parser.add_argument('image_files', metavar='image_file', nargs='*', help='Input image files')
    args = parser.parse_args()

    yolo_model = YOLO(args.pretrained_ultralytics_yolo_model)

    model_class_names = get_ultralytics_model_class_names(yolo_model)
    class_name_to_index = get_ultralytics_class_name_to_index(yolo_model)
    classes = args.classes
    if not classes:
        classes = model_class_names

    invalid_classes = []
    for class_name in classes:
        if class_name not in class_name_to_index:
            invalid_classes.append(class_name)
    if invalid_classes:
        parser.error(
            'Unknown class name(s) for model %s: %s' % (
                args.pretrained_ultralytics_yolo_model,
                ', '.join(invalid_classes),
            )
        )

    try:
        filenames, hwc_bgr_888_ndarrays = load_images_from_paths(args.image_files)
    except RuntimeError as exc:
        parser.error(str(exc))

    app = QApplication(sys.argv)
    window = MainWindow(
        yolo_model=yolo_model,
        yolo_model_path=args.pretrained_ultralytics_yolo_model,
        filenames=filenames,
        hwc_bgr_888_ndarrays=hwc_bgr_888_ndarrays,
        classes=classes,
    )
    window.show()
    sys.exit(exec_qapplication(app))


if __name__ == '__main__':
    main()
