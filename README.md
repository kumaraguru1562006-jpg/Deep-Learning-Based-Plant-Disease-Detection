# 🌿 Deep Learning-Based Plant Disease Detection
### PlantVillage Dataset | MobileNetV2 | Flask Web App

![Python](https://img.shields.io/badge/Python-3.9+-blue) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange) 
![Accuracy](https://img.shields.io/badge/Accuracy-98.2%25-brightgreen)
![Classes](https://img.shields.io/badge/Classes-38-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Introduction

Plant diseases cause significant crop losses worldwide, threatening food security and farmer livelihoods. Early detection is critical but often requires expert knowledge that isn't accessible to smallholder farmers. This project uses **deep learning** and the **PlantVillage dataset** to create an accessible tool that can identify 38 plant disease categories from leaf photographs with over 98% accuracy.

---

## 🎯 Objectives

- Build a high-accuracy CNN model for plant disease classification
- Support 14 plant species and 24 disease types from PlantVillage
- Provide an intuitive web interface for farmers and agricultural workers
- Enable real-time disease detection from smartphone photos
- Offer treatment recommendations alongside predictions

---

## 🗂️ Project Structure

```
plant_disease_detection/
├── app.py                  # Flask web server + API endpoints
├── train.py                # Model training script
├── index.html              # Frontend web application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── dataset/
│   └── PlantVillage/       # Dataset (download separately)
│       ├── Apple___Apple_scab/
│       ├── Tomato___Early_blight/
│       └── ... (38 class folders)
│
└── model/
    ├── plant_disease_model.h5      # Trained model weights
    ├── training_history.json       # Training metrics
    ├── training_history.png        # Accuracy/loss curves
    ├── confusion_matrix.png        # Evaluation confusion matrix
    └── classification_report.json  # Per-class metrics
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Name** | PlantVillage |
| **Total Images** | 54,305 |
| **Classes** | 38 (24 disease + 14 healthy) |
| **Plant Species** | 14 |
| **Image Size** | 224×224 (resized) |
| **Source** | Kaggle / Hughes & Salathé (2015) |

**Supported Plants:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## 🧠 Model Architecture

### MobileNetV2 Transfer Learning

```
Input (224×224×3)
    │
    ▼
Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
    │
    ▼
MobileNetV2 Backbone (pretrained on ImageNet, 2.2M params)
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
BatchNormalization → Dropout(0.3)
    │
    ▼
Dense(512, ReLU)
    │
    ▼
BatchNormalization → Dropout(0.5)
    │
    ▼
Dense(38, Softmax) ← Output
```

**Training Configuration:**
- Optimizer: Adam (lr=1e-4, fine-tune lr=1e-5)
- Loss: Categorical Crossentropy
- Epochs: 15 (Phase 1: 10 frozen, Phase 2: 5 fine-tuned)
- Batch size: 32

---

## 📈 Results

| Metric | Value |
|---|---|
| Training Accuracy | 99.1% |
| Validation Accuracy | 98.2% |
| Top-5 Accuracy | 99.8% |
| Average F1-Score | 0.981 |

---

## 🚀 Setup & Installation

### 1. Clone & Install
```bash
git clone <repo-url>
cd plant_disease_detection
pip install -r requirements.txt
```

### 2. Download Dataset
Download PlantVillage from Kaggle:
```bash
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d dataset/PlantVillage
```

### 3. Train the Model
```bash
python train.py
```

### 4. Run the Web App
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🌐 API Endpoints

### `POST /api/predict`
Upload a leaf image and get disease prediction.

**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "success": true,
  "prediction": {
    "plant": "Tomato",
    "disease": "Early Blight",
    "confidence": 0.923,
    "confidence_percent": "92.3%",
    "is_healthy": false,
    "severity": "Medium",
    "description": "Caused by Alternaria solani fungus...",
    "treatment": "Remove infected leaves. Apply copper fungicides..."
  },
  "top_predictions": [...]
}
```

### `GET /api/classes`
Returns all 38 supported disease classes grouped by plant.

### `GET /api/stats`
Returns dataset statistics and model performance metrics.

### `GET /api/health`
Health check endpoint.

---

## 🔮 Future Scope

1. **Mobile App** — React Native / Flutter with offline inference using TFLite
2. **Real-time Detection** — Webcam feed with live prediction overlay
3. **Expanded Dataset** — Additional crops: wheat, rice, cotton, sugarcane
4. **Severity Scoring** — Quantify disease spread (0–100% leaf area affected)
5. **Geospatial Tracking** — Map disease outbreaks by GPS location
6. **Multi-language Support** — Hindi, Tamil, Swahili, Spanish for global reach
7. **Weather Integration** — Disease risk forecasting based on climate conditions
8. **Prescription Engine** — Automated treatment dosage calculator

---

## 📖 References

1. Hughes, D. & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060
2. Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018
3. PlantVillage Dataset: https://plantvillage.psu.edu/

---

## 📄 License

MIT License — Free for educational and research use.

---

Team members

Kumaraguru v

Mohamed Irfan I

*Built with ❤️ for farmers worldwide*
