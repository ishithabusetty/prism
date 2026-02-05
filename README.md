# Samsung PRISM - Harmful Content Detection Pipeline

> **Real-World AI System for Detecting Harmful Content in Images/Videos**

A production-ready, multi-stage AI pipeline that detects:
- 🔫 **Weapons & Dangerous Objects** (guns, knives, bats, etc.)
- 📝 **Unsafe Text** (promotional, abusive, hate speech)
- 🏷️ **Competitor Brand Logos** (Apple, Google, Huawei, etc.)

**Final Output: SAFE ✅ or UNSAFE 🚨**

---

## 📁 Project Structure

```
prism/
├── 📓 notebooks/                    # Google Colab Notebooks
│   ├── 01_weapon_detection.ipynb    # Phase 1: YOLOv8 training
│   ├── 02_text_classification.ipynb # Phase 2: OCR + DistilBERT
│   ├── 03_logo_detection.ipynb      # Phase 3: Logo YOLOv8
│   └── 04_unified_pipeline.ipynb    # Phase 4: Integrated pipeline
│
├── 📦 src/                          # Source Modules
│   ├── detectors/
│   │   ├── weapon_detector.py       # YOLOv8 weapon detection
│   │   ├── text_detector.py         # EasyOCR + NLP
│   │   └── logo_detector.py         # Logo detection
│   ├── pipeline/
│   │   ├── unified_pipeline.py      # Main inference pipeline
│   │   ├── decision_engine.py       # SAFE/UNSAFE logic
│   │   └── video_processor.py       # Video frame extraction
│   └── utils/
│       ├── data_preparation.py      # Dataset utilities
│       └── visualization.py         # Result visualization
│
├── ⚙️ config/                       # Configuration Files
│   ├── settings.yaml                # Global settings
│   ├── weapon_classes.yaml          # Weapon class definitions
│   ├── logo_classes.yaml            # Logo class definitions
│   └── text_classification.yaml     # NLP configuration
│
├── 📊 data/                         # Datasets (user-provided)
│   ├── weapons/                     # Weapon training data
│   ├── logos/                       # Logo training data
│   └── text_classifier/             # NLP training data
│
├── 🎯 models/                       # Trained Models (after training)
│   ├── weapon_detector/
│   ├── logo_detector/
│   └── text_classifier/
│
├── 📄 requirements.txt              # Python dependencies
└── 📖 README.md                     # This file
```

---

## 🚀 Quick Start (Google Colab)

### Step 1: Upload to Google Drive
Copy this entire `prism/` folder to:
```
Google Drive/samsung_prism/
```

### Step 2: Run Notebooks in Order

| # | Notebook | Description | Time |
|---|----------|-------------|------|
| 1 | `01_weapon_detection.ipynb` | Train weapon detector | ~2 hrs |
| 2 | `02_text_classification.ipynb` | Train text classifier | ~30 min |
| 3 | `03_logo_detection.ipynb` | Train logo detector | ~1 hr |
| 4 | `04_unified_pipeline.ipynb` | Run complete pipeline | ~10 min |

### Step 3: Enable GPU
In each Colab notebook:
- Go to `Runtime` → `Change runtime type` → Select `GPU`

---

## 📥 Dataset Download

### Weapon Detection Datasets
| Dataset | Kaggle Link |
|---------|-------------|
| Guns Detection | [issaisasank/guns-object-detection](https://www.kaggle.com/datasets/issaisasank/guns-object-detection) |
| Knife Detection | [aadityarathod/knife-dataset](https://www.kaggle.com/datasets/aadityarathod/knife-dataset) |
| Violence Situations | [mohamedmustafa/real-life-violence-situations-dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) |

### Logo Detection Datasets
| Dataset | Kaggle Link |
|---------|-------------|
| LogoDet-3K | [lyly99/logodet3k](https://www.kaggle.com/datasets/lyly99/logodet3k) |
| FlickrLogos-32 | [kozodoi/flickrlogos32](https://www.kaggle.com/datasets/kozodoi/flickrlogos32) |

### Dataset Folder Structure
After download, organize as:
```
data/weapons/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 🎯 Detection Classes

### Weapon Classes (9)
| ID | Class |
|----|-------|
| 0 | gun |
| 1 | pistol |
| 2 | rifle |
| 3 | knife |
| 4 | machete |
| 5 | bat |
| 6 | alcohol_bottle |
| 7 | broken_bottle |
| 8 | sword |

### Logo Classes (10)
| ID | Brand | Competitor? |
|----|-------|-------------|
| 0 | apple | ⚠️ Yes |
| 1 | google | ⚠️ Yes |
| 2 | huawei | ⚠️ Yes |
| 3 | xiaomi | ⚠️ Yes |
| 4 | oneplus | ⚠️ Yes |
| 5 | oppo | ⚠️ Yes |
| 6 | vivo | ⚠️ Yes |
| 7 | sony | ⚠️ Yes |
| 8 | lg | ⚠️ Yes |
| 9 | samsung | ✅ Own |

### Text Classification Labels
| Label | Description |
|-------|-------------|
| 🟢 SAFE | Normal, harmless text |
| 🟡 PROMOTIONAL | Sales, discounts, marketing |
| 🔴 ABUSIVE | Hate speech, threats |

---

## ⚖️ Decision Logic

```
IF weapon detected (confidence ≥ 50%) → UNSAFE
ELSE IF abusive text detected (confidence ≥ 80%) → UNSAFE  
ELSE IF promotional text detected (confidence ≥ 70%) → UNSAFE
ELSE IF competitor logo detected (confidence ≥ 60%) → UNSAFE
ELSE → SAFE
```

---

## 📋 Sample Output

```json
{
  "status": "UNSAFE",
  "is_safe": false,
  "summary": "UNSAFE: Weapon detected (gun) - Confidence: 92.50%",
  "flags": [
    {
      "type": "WEAPON_DETECTED",
      "priority": 1,
      "confidence": 0.925,
      "objects": ["gun"]
    }
  ],
  "detailed_results": {
    "weapon_detection": {
      "detected": true,
      "detections": [
        {"class": "gun", "confidence": 0.925, "bbox": {...}}
      ]
    },
    "text_classification": {
      "text_found": false
    },
    "logo_detection": {
      "competitor_detected": false
    }
  }
}
```

---

## 🔧 Local Setup (Optional)

If running locally instead of Colab:

```bash
# Clone/copy project to F:\proj\prism
cd F:\proj\prism

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run training (requires GPU)
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

---

## 📊 Expected Metrics

| Model | Metric | Target |
|-------|--------|--------|
| Weapon Detector | mAP@50 | ≥ 0.85 |
| Logo Detector | mAP@50 | ≥ 0.80 |
| Text Classifier | Accuracy | ≥ 0.90 |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Object Detection | YOLOv8 (Ultralytics) |
| OCR | EasyOCR |
| NLP | DistilBERT (HuggingFace) |
| Deep Learning | PyTorch |
| Environment | Google Colab (GPU) |

---

## 📝 License

Samsung PRISM Project - Internal Use

---

## 👥 Contributors

Samsung PRISM Team

---

## 📧 Support

For issues or questions, contact the project team.
