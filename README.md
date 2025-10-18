# Skin Disease Image Classifier

This project uses **TensorFlow/Keras** to classify images of skin into different categories such as Acne, Dry Skin, Wrinkles, and more. It can handle single image predictions or batch predictions from a dataset.
## Dataset source : 
https://universe.roboflow.com/parin-kittipongdaja-vwmn3/skin-problem-multilabel/dataset/1 
## Features

* Classifies multiple skin problems from images.
* Uses **MobileNet** as the base model with a custom head.
* Includes data preprocessing, augmentation, and class balancing.
* Saves trained models for reuse.

## Setup

1. Clone this repository:

```bash
git clone <your_repo_url>
cd <repo_name>
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Place the trained model in the `models` folder (or download if needed):

```bash
models/best_head_only.keras
```

## Usage

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("models/best_head_only.keras")

# Load and preprocess an image
img = image.load_img("path_to_image.jpg", target_size=(224,224))
# Preprocess & predict as in the notebook
```

## Notes

* For best performance, run on **GPU**.
* Preprocessing must match the training preprocessing (MobileNet).
* You can expand the dataset and retrain the model for better accuracy.


