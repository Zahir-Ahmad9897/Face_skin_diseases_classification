# 🧴 AI Skin Disease Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

An AI-powered web application that classifies facial skin conditions into **10 different categories** using deep learning. This project uses a **MobileNet** architecture trained on a dataset of skin issues to provide real-time predictions with confidence scores.

---

## 🚀 Features

- **Real-time Classification**: Upload an image and get instant predictions.
- **Top-k Visualization**: Displays the top 3 most likely conditions with confidence bars.
- **MobileNet Architecture**: Lightweight and efficient model optimized for speed.
- **Privacy-Focused**: Images are processed locally in your session and not saved.

## 🧠 Model Classes

The model is trained to detect the following 10 skin conditions:

1.  **Acne**
2.  **Blackheads**
3.  **Dark Spots**
4.  **Dry Skin**
5.  **Eye Bags**
6.  **Normal Skin**
7.  **Oily Skin**
8.  **Pores**
9.  **Skin Redness**
10. **Wrinkles**

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- [Git LFS](https://git-lfs.com/) (Required for downloading large model files)

### 1. Clone the Repository

```bash
git clone https://github.com/Start-code143/Face_skin_diseases_classification.git
cd Face_skin_diseases_classification
```

### 2. Download the Model

> **Note:** The model file (`best_head_only.keras`) is large. If it wasn't downloaded automatically by Git LFS, please download it manually from the releases or your backup and place it in the `model/` directory (or root).

### 3. Install Dependencies

It is recommended to use a virtual environment.

```bash
# Create virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## 📂 Project Structure

```
├── .gitignore              # Files to exclude from Git
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── Face_diseases_classification.ipynb  # Training notebook (Colab)
├── model/                  # Directory for trained models
│   └── best_head_only.keras
└── assets/                 # Images and logos
    └── samples/            # Sample images for testing
```

## 📊 Dataset

The model was trained on the **Skin Problem Multi-Label Dataset**:
- **Source**: [Roboflow Universe](https://universe.roboflow.com/parin-kittipongdaja-vwmn3/skin-problem-multilabel/dataset/1)
- **Features**: Handling of imbalanced data, augmentation, and MobileNet preprocessing.

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.** It is **not** a diagnostic tool. If you have concerns about your skin health, please consult a certified dermatologist.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1.  Fork the repo
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
