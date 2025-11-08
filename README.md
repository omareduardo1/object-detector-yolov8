# Real-Time Object Detector (YOLOv8 + OpenCV)

A **real-time computer vision** project based on **YOLOv8** and **OpenCV**, developed for automatic object recognition through webcam or video streams.  
Fully configured for **Visual Studio Code** and optimized for **macOS Apple Silicon (MPS)**, but also works on **CPU** and **CUDA GPUs**.

---
## YOLOv8 Example — “Bus” Detection Demo

Below is an example of YOLOv8 detecting multiple objects from a single image (`bus.jpg`):  
- **Bus**  
- **Persons**  
- **Traffic lights**  
- **Cars**  

It demonstrates YOLO’s ability to recognize several object categories simultaneously.

<p align="center">
  <img src="/Users/omarborges/Desktop/Progettini GitHub/Real-Time Object Detector/runs/detect/predict/bus.jpg" alt="YOLOv8 Bus Detection Example" width="720">
</p>

To reproduce this image yourself, run:
```bash
yolo predict model=yolov8n.pt source='/Users/omarborges/Desktop/Progettini GitHub/Real-Time Object Detector/bus.jpg' conf=0.25 imgsz=640
---

## Project Description

This program performs **real-time object detection** using a pretrained **YOLOv8 model** (trained on the **COCO dataset**, 80 common classes).  
It captures frames from your webcam, processes them through YOLOv8, and overlays **bounding boxes, labels, and FPS** directly on the live video feed.

Designed to be:
- **Educational** – to understand how real-time AI vision works  
- **Modular** – easy to extend with new models or datasets  
- **Optimized** – runs smoothly even on MacBooks with M1/M2 chips  

---

## Main Features
- **Real-time object detection** from webcam or video file (`--cam 0` or `--cam video.mp4`)  
- **YOLOv8 pretrained model** with 80 COCO classes  
- **Multi-device support**: `--device mps|cpu|cuda`  
- **Dynamic confidence control** (+ / - while running)  
- **FPS counter** overlay  
- **Bounding box & label visualization**  
- **Ready for custom training** on your own datasets  

---

## Project Structure
object-detector-yolov8/
│
├── src/
│   └── infer_webcam.py          # Main script
│
├── .vscode/                     # VS Code configurations
│
├── requirements.txt             # Required libraries
├── .gitignore                   # Ignored files
├── README.md                    # (this file)
└── models/ (optional)           # YOLO weights (not tracked)

---

## Quick Start (VS Code)

1. Open the folder in **Visual Studio Code**  
2. Install the **Python** extension (Microsoft)  
3. Press **F5** to run (VS Code will create `.venv` and install dependencies)  
4. A webcam window will open:
   - Press **ESC** to exit  
   - Press **+ / -** to adjust confidence threshold  

---

## Run from Terminal

### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/infer_webcam.py --device mps --imgsz 416 --conf 0.5 --max_det 50 --cam 0