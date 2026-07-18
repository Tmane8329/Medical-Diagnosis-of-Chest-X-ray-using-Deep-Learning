import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained model

# Replace your current load_model line with this:
model = tf.keras.models.load_model(
    "models/best_model.h5",
    compile=False,
    safe_mode=False  # This helps bypass some Keras 3 validation checks
)

# 14 labels
labels = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia"
]

# Load test image (change name to any image from your dataset)
img_path = "data/images/00000001_000.png"

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
predictions = model.predict(img_array)[0]

print("\n🔍 Prediction Results:\n")

for i in range(len(labels)):
    print(f"{labels[i]}: {round(predictions[i]*100, 2)}%")
