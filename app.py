 

import os
import textwrap
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------
# TensorFlow / Keras
# ---------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.mobilenet import preprocess_input

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ---------------------
# Page configuration
# ---------------------
st.set_page_config(
    page_title="AI Skin Classifier",
    page_icon="👩‍⚕️",
    layout="wide",
)

# ---------------------
# Constants
# ---------------------
DEFAULT_CLASS_NAMES: list[str] = [
    "Acne",
    "Blackheads",
    "Dark Spots",
    "Dry Skin",
    "Eye bags",
    "Normal Skin",
    "Oily Skin",
    "Pores",
    "Skin Redness",
    "Wrinkles",
]

DEFAULT_MODEL_PATHS: list[str] = [
    "./model/best_head_only.keras",
    "./model/best_head_only.h5",
    "./best_head_only.keras",
    "./best_head_only.h5",
]

ASSETS_DIR = Path("assets")
SAMPLES_DIR = ASSETS_DIR / "samples"
TARGET_SIZE = (224, 224)


# ---------------------
# Helper utilities
# ---------------------
@st.cache_resource
def load_keras_model(paths_to_try: list[str] | None = None):
    """
    Attempt to load a Keras model from several candidate paths.

    Returns
    -------
    (model, info) where *info* is the path string on success,
    ``"tensorflow_import_error"`` when TF is missing, or the last
    exception on failure.
    """
    if not TF_AVAILABLE:
        return None, "tensorflow_import_error"

    if paths_to_try is None:
        paths_to_try = DEFAULT_MODEL_PATHS

    last_error = None
    for path in paths_to_try:
        path = str(path).strip()
        if os.path.isfile(path):
            try:
                model = load_model(path)
                return model, path
            except Exception as exc:
                last_error = exc
    return None, last_error


def preprocess_pil(img_pil: Image.Image, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Resize, convert to array, and apply MobileNet preprocessing."""
    img = img_pil.convert("RGB").resize(target_size)
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def confidence_percent(value: float) -> float:
    """Convert a probability (0‑1) to a percentage."""
    return float(value * 100)


# ---------------------
# Custom CSS
# ---------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f7f9ff 0%, #fef6fb 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f7f9ff 0%, #fef6fb 100%);
    }
    .prediction-box {
        border-radius: 12px;
        padding: 14px;
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------
# Header
# ---------------------
ASSETS_DIR.mkdir(exist_ok=True)
logo_path = ASSETS_DIR / "logo.png"

col_logo, col_title = st.columns([1, 5])
with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), width=110)
    else:
        st.markdown("### 👩‍⚕️\n**Skin Classifier**")
with col_title:
    st.markdown("## AI-Based Skin Problem Classifier")
    st.markdown(
        "Upload a clear facial/skin photo and the model will predict the most "
        "likely skin conditions. This app shows top predictions and confidence. "
        "⚠️ Not a medical diagnosis — consult a dermatologist."
    )

st.divider()

# ---------------------
# Class names (editable)
# ---------------------
CLASS_NAMES = list(DEFAULT_CLASS_NAMES)

# ---------------------
# Sidebar — model loading
# ---------------------
st.sidebar.header("Model Options")
st.sidebar.markdown("Choose how to load the model:")

load_mode = st.sidebar.radio(
    "Load mode",
    ("Auto (try default paths)", "Upload model file (.keras/.h5)"),
)

model = None
model_path_used = None

if load_mode == "Auto (try default paths)":
    with st.sidebar.expander("Paths tried (editable)"):
        user_paths = st.text_area(
            "One path per line:",
            value="\n".join(DEFAULT_MODEL_PATHS),
            height=120,
        )
    paths_list = [p.strip() for p in user_paths.splitlines() if p.strip()]

    with st.spinner("Loading model (auto)…"):
        model, model_path_used = load_keras_model(paths_list)
        if model is None:
            if model_path_used == "tensorflow_import_error":
                st.sidebar.error(
                    "TensorFlow import failed. Make sure `tensorflow` is installed."
                )
            else:
                st.sidebar.warning(
                    "Model not found in default paths. Switch to "
                    "**Upload model file** to upload a model directly."
                )
        else:
            st.sidebar.success(f"Model loaded from: `{model_path_used}`")
else:
    uploaded_model_file = st.sidebar.file_uploader(
        "Upload a Keras model (.keras or .h5)", type=["keras", "h5"]
    )
    if uploaded_model_file is not None:
        model_dir = Path("model")
        model_dir.mkdir(exist_ok=True)
        tmp_path = model_dir / uploaded_model_file.name
        tmp_path.write_bytes(uploaded_model_file.getbuffer())

        with st.spinner("Loading uploaded model…"):
            try:
                model = load_model(str(tmp_path))
                model_path_used = str(tmp_path)
                st.sidebar.success(f"Model loaded: **{tmp_path.name}**")
            except Exception as exc:
                st.sidebar.error(f"Failed to load uploaded model: {exc}")

if model is None:
    st.warning(
        "Model is not loaded. Place the model in `./model/` or upload it "
        "via the sidebar."
    )
    st.info(
        "If your model is on Google Drive (Colab), download it locally to "
        "the repo's `model/` folder, or upload it here."
    )

# ---------------------
# Main area — image input
# ---------------------
col_img, col_ctrl = st.columns([1.2, 1])

with col_img:
    st.markdown("### 📸 Upload or Try Sample Images")
    uploaded_file = st.file_uploader(
        "Upload an image (jpg / png)", type=["jpg", "jpeg", "png"]
    )

    st.markdown("**OR** select a sample image (if any in `assets/samples/`):")
    sample_files = sorted(SAMPLES_DIR.glob("*.*")) if SAMPLES_DIR.exists() else []
    sample_choice = None

    if sample_files:
        sample_names = [p.name for p in sample_files]
        sample_choice = st.selectbox("Choose sample", ["-- none --"] + sample_names)
    else:
        st.info("No samples found. Add images to `assets/samples/` for quick tests.")

    # Determine the active image
    pil_img = None
    if uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file)
        except Exception as exc:
            st.error(f"Cannot open uploaded file: {exc}")
    elif sample_choice and sample_choice != "-- none --":
        pil_img = Image.open(SAMPLES_DIR / sample_choice)

    if pil_img is not None:
        st.image(pil_img, caption="Input image", use_container_width=True)
    else:
        st.markdown("_No image loaded yet. Upload or select a sample to run prediction._")

with col_ctrl:
    st.markdown("### 🔧 Prediction Settings")
    top_k = st.slider("Top predictions to show", min_value=1, max_value=5, value=3)
    run_button = st.button("Run Prediction", type="primary")

# Advanced: class-name editing
with st.expander("Advanced: Edit class names (must match model output order)"):
    text_classes = st.text_area(
        "One class per line (order matters)",
        value="\n".join(CLASS_NAMES),
        height=190,
    )
    edited = [s.strip() for s in text_classes.splitlines() if s.strip()]
    if edited:
        CLASS_NAMES = edited

# ---------------------
# Prediction
# ---------------------
if run_button:
    if pil_img is None:
        st.error("Please upload or select an image first.")
    elif model is None:
        st.error(
            "Model not loaded. Upload a model file in the sidebar or place "
            "it in `./model/`."
        )
    else:
        try:
            with st.spinner("Preprocessing image and predicting…"):
                x = preprocess_pil(pil_img)
                preds = model.predict(x)[0]

                n_out = len(preds)
                if n_out != len(CLASS_NAMES):
                    st.warning(
                        f"Model output length ({n_out}) ≠ number of class names "
                        f"({len(CLASS_NAMES)}). Using first "
                        f"{min(n_out, len(CLASS_NAMES))} classes."
                    )

                top_indices = np.argsort(preds)[-top_k:][::-1]
                st.markdown("### 🩺 Prediction Results")
                st.markdown(
                    "<div class='prediction-box'>", unsafe_allow_html=True
                )
                for idx in top_indices:
                    class_name = (
                        CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
                    )
                    pct = confidence_percent(preds[idx])
                    st.write(f"**{class_name}** — {pct:.2f}%")
                    st.progress(int(np.clip(pct, 0, 100)))
                st.markdown("</div>", unsafe_allow_html=True)

                st.info(
                    "💡 This is a trained-model output. For medical advice, "
                    "consult a professional."
                )
        except Exception as exc:
            st.error(f"Error during prediction: {exc}")

# ---------------------
# Footer
# ---------------------
st.divider()
st.markdown(
    """
**Notes & Troubleshooting**
- Place the model file inside `./model/` or upload via the sidebar.
- If you used different preprocessing during training (not MobileNet),
  update the `preprocess_pil` function accordingly.
- Model files can be large — use
  [Git LFS](https://git-lfs.github.com/) for files > 100 MB.
"""
)
