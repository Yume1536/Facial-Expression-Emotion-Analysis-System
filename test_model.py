import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("FER_MODEL_PATH", BASE_DIR / "outputs" / "models" / "CNNModel_fer_3emo.h5"))
TEST_DIR = Path(os.getenv("FER_TEST_DIR", BASE_DIR / "inputs" / "fer"))
OUTPUT_DIR = Path(os.getenv("FER_OUTPUT_DIR", BASE_DIR / "outputs" / "confusion_matrix"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["Angry", "Happy", "Surprise"]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not TEST_DIR.exists():
    raise FileNotFoundError(f"Dataset directory not found: {TEST_DIR}")

print("Loading model...")
model = load_model(str(MODEL_PATH), compile=False)

print("Loading test data...")
X_test, y_test = [], []
for idx, emotion in enumerate(EMOTIONS):
    folder = TEST_DIR / emotion
    if not folder.exists():
        print(f"Warning: folder not found - {folder}")
        continue

    for img_file in folder.iterdir():
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (48, 48))
        img = img.astype("float32") / 255.0
        X_test.append(img)
        y_test.append(idx)

if not X_test:
    raise ValueError("No test images were loaded. Check FER_TEST_DIR and class folders.")

X_test = np.expand_dims(np.array(X_test), -1)
y_test = np.array(y_test)
print(f"Loaded {len(X_test)} test samples")

print("Evaluating model...")
y_pred = np.argmax(model.predict(X_test), axis=1)
report = classification_report(y_test, y_pred, target_names=EMOTIONS)
print("\nClassification Report:")
print(report)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=EMOTIONS, yticklabels=EMOTIONS)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
cm_path = OUTPUT_DIR / "conf_matrix.png"
plt.savefig(cm_path)
plt.close()
print(f"\nConfusion matrix saved to {cm_path}")
