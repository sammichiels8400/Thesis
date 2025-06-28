import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import wandb
from itertools import combinations

df = pd.read_csv("../../data/csv/cleaned_fix.csv")

columns = ["Image", "Scene", "ROI", "FixDur", "FixNr", "Start", "End", "X", "Y", "Name", "NextROI"]
feature_sets = []
for r in range(1, len(columns)+1):
    feature_sets.extend(list(combinations(columns, r)))

def train():
    run = wandb.init()
    config = run.config

    selected_features = feature_sets[config.feature_set_idx]
    X = df[selected_features]
    y = df.iloc[:, -1]   # last column as target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_numeric = X.select_dtypes(include=['int64', 'float64'])  # Keep only numeric columns
    X_scaled = StandardScaler().fit_transform(X_numeric)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(9,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
    
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
    
        layers.Dense(32, activation='relu'),
        
        layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_split=0.1)

    wandb.log({
        "val_accuracy": history.history['val_accuracy'][-1],
        "val_loss": history.history['val_loss'][-1],
        "features": selected_features
    })