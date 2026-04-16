# `YOLOroom`

A Qt-based application for reviewing and exporting Ultralytics YOLO object detection results on images within a
Photoshop/Lightroom-inspired interface.

![YOLOroom](https://github.com/jifengwu2k/yoloroom/blob/main/SCREENSHOT.png?raw=true)

## Features

- **Ultralytics YOLO detection:** Loads a pretrained Ultralytics YOLO model and runs object detection on images provided
  at startup or opened later from the UI.
- **Broad Qt binding support:** Works with `PyQt6`, `PySide6`, `PyQt5`, `PySide2`, `PyQt4`, and `PySide` via
  `detect-qt-binding`.
- **Photo-Centric UI:**
    - **Bottom Thumbnail Strip** shows thumbnails of all loaded images for quick navigation.
    - **Central Display** shows the current image with detection annotations.
    - **Right Sidebar** is split into two sections:
        - **Open Images Button**
            - Lets you open one or more additional images after startup.
            - Makes it possible to launch `YOLOroom` with no images and load them later.
        - **Close Current Image Button**
            - Removes the currently selected image from the session.
            - Works even when it is the last remaining image, returning the UI to the empty state.
        - **Export Panel**
            - Used to export each checked instance as an individual image.
            - Specify export directory and image format (JPEG/PNG).
            - Specify image name template.
                - Supports the following variables:
                    - `filename`: The original image file name.
                    - `class`: The class of the instance (for example `person`, `bicycle`).
                    - `index`: The index of the instance (for example `0`, `1`).
        - **Classes and Instances Panel**
            - Lists detected classes as groups.
            - Each class has:
                - An assigned color used for annotation visualization.
                - A checkbox to toggle all instances of this class.
                - A list of instances.
                    - Each instance has its own checkbox.
                    - Each instance shows its confidence score.

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install one supported Qt binding separately, for example:

```bash
pip install PyQt6
```

## Usage

```bash
python yoloroom.py [--pretrained-ultralytics-yolo-model MODEL] [--class CLASS ...] [<image_file> ...]
```

Examples:

```bash
python yoloroom.py
python yoloroom.py image1.jpg image2.jpg
python yoloroom.py --pretrained-ultralytics-yolo-model yolo26n.pt image1.jpg
python yoloroom.py --class person --class car image1.jpg image2.jpg
```

### CLI options

- `--pretrained-ultralytics-yolo-model`
    - Pretrained Ultralytics YOLO model name or path.
    - Default: `yolo26n.pt`
- `--class`
    - Optional class filter.
    - Repeat the flag to include multiple classes.
    - If omitted, all classes supported by the selected model are enabled.

## A New Kind of Computer Vision Tool

Most tools for computer vision treat images as data, leaving ML practitioners and creative professionals to work
through spreadsheet-like dashboards or simplistic box editors. But what if working with detections felt as intuitive and
powerful as working in Photoshop or Lightroom?

`YOLOroom` is built on a simple idea: reviewing and exporting model results should feel as natural and expressive as
creative editing.
