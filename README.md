# 🎯 Face Detector System

A real-time face detection and tracking system built with **YOLOv3-face**, capable of detecting and capturing distinct faces from images, videos, or live webcam feeds.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📖 Overview

This project implements a robust face detection system that can:

- **Detect faces** in real-time using YOLOv3 deep learning model
- **Track distinct faces** using IOU-based position tracking
- **Capture high-quality samples** of each unique face detected
- **Save distinct face images** automatically to disk
- Process **images**, **videos**, and **live webcam** feeds

Built on top of the [YOLOFace](https://github.com/sthanhng/yoloface) repository by **sthanhng**, this system extends the original implementation with face tracking, quality scoring, and a modern desktop GUI.

---

## 🚀 Applications

| Application | Description |
|-------------|-------------|
| 📋 **Attendance Monitoring** | Automatically capture and log faces for attendance tracking in schools, offices, or events |
| 🔐 **Security Systems** | Monitor and record distinct individuals entering restricted areas |
| 👥 **Crowd Analysis** | Count and track unique faces in crowded environments |
| 🏢 **Access Control** | Identify and log visitors in buildings or facilities |
| 📊 **Customer Analytics** | Track unique customers in retail environments |

---

## ✨ Features

- 🎯 **Real-time Detection** - Fast face detection using YOLOv3 architecture
- 🔍 **Distinct Face Tracking** - Tracks faces across frames to avoid duplicates
- 📸 **Quality Scoring** - Automatically selects the best quality image of each face
- 💾 **Auto-save** - Saves detected faces to disk immediately
- 🖥️ **Desktop GUI** - Modern PyQt5 interface for easy use
- ⚙️ **Configurable Settings** - Adjust tolerance, skip frames, and more
- 📹 **Multiple Sources** - Support for images, videos, and webcam

---

## 🛠️ Installation & Usage


### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/youssefmmagdy/Face-Detector.git
   cd Face-Detector
   ```

2. **Run the setup and launch script**
   ```bash
   ./setup_and_run.sh
   ```

   This script will automatically:
   - Install all required dependencies
   - Download the model weights
   - Launch the desktop application



### Build Standalone Executable (Windows)

To create a standalone .exe file (no terminal window), run:

```bash
pyinstaller --onefile --noconsole --name yoloface_app --add-data "cfg;cfg" --add-data "model-weights;model-weights" yoloface_app/main.py --noconfirm
```

The executable will be created in the `dist/` folder as `yoloface_app.exe`.

---

### Desktop Application Features

- Select source (Image / Video / Webcam)
- Real-time preview with detection boxes
- View and manage detected faces
- Configurable settings sidebar

---

## 📁 Project Structure

```
Face-Detector/
├── yoloface_app/           # Desktop GUI application
│   ├── core/               # Detection worker and logic
│   └── ui/                 # PyQt5 UI components
├── cfg/                    # YOLO configuration files
│   ├── yolov3-face.cfg     # Network architecture
│   └── face.names          # Class names
├── model-weights/          # Model weights (download required)
├── samples/                # Sample images/videos
├── outputs/                # Detection outputs
│   └── distinct_faces/     # Saved distinct face images
├── yoloface.py             # CLI face detection
├── yoloface_tracker.py     # CLI face tracking
├── utils.py                # Utility functions
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tolerance` | Face matching threshold (0-1) | 0.6 |
| `skip_frames` | Process every Nth frame | 10 |
| `sample_timeout` | Seconds before saving best face | 3 |
| `save_faces` | Save distinct face images | True |
| `save_output` | Save annotated video/image | True |

---

## 🙏 Acknowledgments

- **[YOLOFace](https://github.com/sthanhng/yoloface)** by **sthanhng** - Original YOLOv3-face implementation
- **[YOLO](https://pjreddie.com/darknet/yolo/)** by Joseph Redmon - YOLO object detection
- **[WIDER FACE](http://shuoyang1213.me/WIDERFACE/)** - Training dataset

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Youssef M. Magdy**

- GitHub: [@youssefmmagdy](https://github.com/youssefmmagdy)

---

<p align="center">
  <i>Built with ❤️ for computer vision enthusiasts</i>
</p>
