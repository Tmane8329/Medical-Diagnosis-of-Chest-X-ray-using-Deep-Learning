import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# =========================
# 1. Load CSV
# =========================
df = pd.read_csv("data/Data_Entry_2017.csv")

# 14 Disease Labels
labels = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia"
]

# Convert multi-label string into binary columns
for label in labels:
    df[label] = df["Finding Labels"].apply(lambda x: 1 if label in x else 0)

# Keep only required columns
df = df[["Image Index"] + labels]

# =========================
# 2. Image Data Generator
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory="data/images",
    x_col="Image Index",
    y_col=labels,
    target_size=(224, 224),
    batch_size=16,
    class_mode="raw",
    subset="training"
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory="data/images",
    x_col="Image Index",
    y_col=labels,
    target_size=(224, 224),
    batch_size=16,
    class_mode="raw",
    subset="validation"
)

# =========================
# 3. Build Model
# =========================
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False


x = GlobalAveragePooling2D()(base_model.output)
output = Dense(14, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 4. Callbacks
# =========================
checkpoint = ModelCheckpoint(
    "models/best_model.h5",
    monitor="val_loss",
    save_best_only=True
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# =========================
# 5. Train Model
# =========================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[checkpoint, early_stop]
)

print("Training Complete ✅")
