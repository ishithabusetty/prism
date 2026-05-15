# 🛡️ PRISM Vision Intelligence Suite

> **AI-Powered Multi-Modal Detection Framework for Brand Logo Recognition and Risk Object Surveillance**

PRISM Vision Intelligence Suite is a deep-learning-based computer vision framework designed for **real-time logo detection**, **brand intelligence**, and **risk-object surveillance** across images and videos. The project combines **YOLOv8 object detection**, **OCR-based text verification**, and **context-aware risk analysis** to deliver highly reliable multimedia auditing and monitoring.

The repository contains complete **training** and **inference** pipelines for:

- 📱 **Logo Detection**
- 🚨 **Risk & Restricted Object Detection**

Both systems are optimized for **Google Colab GPU execution** and support scalable inference workflows for real-world deployment.

---

# 🚀 Project Modules

## 📱 1. Logo Detection System

A strict multi-modal logo intelligence engine capable of identifying electronic/mobile brand logos using:

- 🎯 YOLOv8 visual object detection
- 🔤 EasyOCR + Tesseract OCR fusion
- 🧠 Brand-aware keyword validation
- ✅ Strict confidence and context filtering

The system performs robust logo verification even in:
- low-resolution media
- partially visible logos
- OCR-only text appearances
- noisy or cluttered frames

---

## 🚨 2. Risk Detection System

An AI-powered surveillance framework for detecting:

- 🔫 Firearms
- 💣 Ammunition
- 🔪 Knives / blades
- 🍺 Alcohol
- 🚬 Tobacco / vape products

The detector additionally performs:

- 🧠 Context-aware reasoning
- 👥 Benign social/kitchen environment filtering
- ⚠️ Dynamic risk-level classification
- 🎥 Timestamped video inference

---

# ✨ Features

## 📱 Logo Detection Features

- Multi-modal logo verification (Visual + OCR fusion)
- Strict false-positive suppression logic
- OCR keyword normalization and validation
- Support for video and image inference
- Brand-specific confidence thresholds
- COCO contextual filtering
- Automatic annotation rendering
- GPU-accelerated inference

---

## 🚨 Risk Detection Features

- Real-time risk object surveillance
- Multi-class YOLOv8 detection pipeline
- Context-sensitive risk downgrading
- Social/kitchen environment awareness
- Timestamp-aware video analysis
- Dynamic severity classification
- Benign context suppression
- Detection summaries and logs

---

# 🧠 System Architecture

```text
Input Media
     │
     ▼
Frame Extraction / Image Input
     │
     ├──────────────┐
     ▼              ▼
YOLOv8          OCR Engine
Detection       (EasyOCR + Tesseract)
     │              │
     └──────┬───────┘
            ▼
    Fusion & Validation
            │
            ▼
 Context / Risk Analysis
            │
            ▼
 Annotated Output + Reports
```

---

# 🛠️ Tech Stack

| Category | Technology |
|---|---|
| 🧠 Detection Model | YOLOv8 (Ultralytics) |
| 🔤 OCR Engine | EasyOCR + Tesseract OCR |
| 📹 Video Processing | OpenCV |
| 📊 Data Handling | Pandas, NumPy |
| 📈 Visualization | Matplotlib |
| 🧪 ML Utilities | Scikit-learn |
| ☁️ Runtime | Google Colab |
| ⚡ GPU Support | NVIDIA T4 GPU |
| 📂 Dataset Handling | Kaggle API + Roboflow |
| 🧾 Configuration | YAML / JSON |

---

# 📂 Repository Structure

```text
.
├── logo_detection_training_colab.ipynb
├── logo_detection_inference_strict_v4_1_colab.ipynb
├── risk_detection_training_fresh_colab.ipynb
├── risk_detection_inference_fresh_colab.ipynb
```

---

# 📓 Notebook Descriptions

| Notebook | Purpose |
|---|---|
| `logo_detection_training_colab.ipynb` | Training pipeline for logo detection using YOLOv8 |
| `logo_detection_inference_strict_v4_1_colab.ipynb` | Strict OCR + visual fusion inference pipeline |
| `risk_detection_training_fresh_colab.ipynb` | Training pipeline for multi-class risk detection |
| `risk_detection_inference_fresh_colab.ipynb` | Real-time risk inference with context filtering |

---

# 📊 Datasets Used

## 📱 Logo Detection Datasets

| Dataset | Source |
|---|---|
| LogoDet-3K | Kaggle |
| Custom Mobile Brand Logo Dataset | Kaggle |
| Electronics & Smartphone Logos | Mixed Sources |

### Supported Logo Brands

| Brands |
|---|
| Motorola |
| Samsung |
| Oppo |
| Vivo |
| HTC |
| Sony |
| Nokia |
| Honor |
| Huawei |
| Asus |
| LG |
| OnePlus |
| Apple |
| Micromax |
| Lenovo |
| Gionee |
| InFocus |
| Tenor |
| Lava |
| Panasonic |
| Intex |
| Jio |
| Reliance |
| Blackberry |
| Google Pixel |

---

## 🚨 Risk Detection Datasets

| Dataset Name | Source |
|---|---|
| weapon-detection-gx69u-5woyh | Roboflow |
| knife-gmyes-kk1b3 | Roboflow |
| cigarette-alcohol-1vbqg | Roboflow |
| alcohol-sxlfn-mlsw1 | Roboflow |
| snehilsanyal/weapon-detection-test | Kaggle |
| iqmansingh/guns-knives-object-detection | Kaggle |
| atulyakumar98/gundetection | Kaggle |
| ankan1998/weapon-detection-dataset | Kaggle |
| prajjwalkumarpanzade/smoking-and-drinking-dataset-for-yolo | Kaggle |
| alihassanml/yolo-dataset-smoking-eating-sleeping-phone | Kaggle |

---

# ⚙️ Installation

## Install Dependencies

```bash
pip install ultralytics
pip install easyocr pytesseract
pip install opencv-python-headless
pip install pandas matplotlib tqdm
pip install pyyaml numpy
```

---

# ☁️ Google Colab Setup

## Recommended Runtime

> ⚡ Use **T4 GPU** for optimal training and inference performance.

### Steps

1. Open notebook in Google Colab
2. Go to:
   ```text
   Runtime → Change Runtime Type
   ```
3. Select:
   ```text
   GPU → T4
   ```

---

# 🔑 Credentials Setup

## Kaggle API

Place your Kaggle credentials:

```json
{
  "username": "YOUR_USERNAME",
  "key": "YOUR_KEY"
}
```

inside:

```text
/root/.kaggle/kaggle.json
```

---

## Roboflow API

Add:

```python
ROBOFLOW_API_KEY = "YOUR_API_KEY"
```

inside the training notebook.

---

# 🎯 Detection Classes

## 📱 Logo Detection Classes

```text
motorola
samsung
oppo
vivo
sony
nokia
apple
oneplus
google
...
```

---

## 🚨 Risk Detection Classes

```text
firearm
ammo
knife_blade
alcohol
tobacco_vape
```

---

# 🧪 Inference Pipeline

## 📱 Logo Detection

- Input image/video
- YOLO logo localization
- OCR text extraction
- Brand keyword matching
- Confidence verification
- Final annotation rendering

---

## 🚨 Risk Detection

- Frame extraction
- Risk object detection
- Context-object detection
- Benign environment filtering
- Risk severity estimation
- Timestamp logging

---

# 📈 Output Features

## Generated Outputs

- ✅ Annotated videos
- ✅ Detection reports
- ✅ Confidence scores
- ✅ Risk classifications
- ✅ Timestamp logs
- ✅ OCR extracted text
- ✅ Detection summaries

---

# 🔥 Key Highlights

> ✅ Multi-modal AI verification  
> ✅ OCR + Vision Fusion  
> ✅ Context-aware intelligence  
> ✅ False-positive suppression  
> ✅ Real-time GPU inference  
> ✅ Video + image support  
> ✅ Scalable Colab workflow  

---

# 🚀 Future Enhancements

- 🌐 Live CCTV integration
- 📡 Real-time streaming support
- 🧠 LLM-powered scene understanding
- 📊 Web dashboard deployment
- ☁️ Cloud inference APIs
- 📱 Mobile deployment support

---

# 📜 License

This repository is intended for educational, research, and prototype development purposes.

---

# 👨‍💻 Author

Developed as part of an AI-powered multimedia intelligence and surveillance research initiative using deep learning and multimodal computer vision workflows.
