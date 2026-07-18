import numpy as np
from PIL import Image

def load_and_preprocess(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return img
