import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# decode numeric prediction into original string
def decode_class(predcition):
    return "MmMmM" if predcition == 1 else "QqQqQ"

# get path to input .csv file from command-line
csv_path = sys.argv[1]

# load trained model and scailing params
model = load_model("model.h5")
scaler_data = np.load("scaler.npz")
scaler_mean = scaler_data["mean"]
scaler_scale = scaler_data["scale"]

# load input data
df = pd.read_csv(csv_path)

# extract features if class column is present, drop it
X = df.drop("class", axis=1).values if "class" in df.columns else df.values

# scale input features using prevously saved scaler
X_scaled = (X - scaler_mean) / scaler_scale

# predict provavilitie and convert to binaty class labels
pred_probs = model.predict(X_scaled)
pred_classes = (pred_probs > 0.5).astype(int).flatten()

# if original class labels are present, compare them
if "class" in df.columns:
    expected_classes = df["class"].map({"QqQqQ": 0, "MmMmM": 1}).values
    correct = 0
    total = len(expected_classes)
    for predcition, expected_text in zip(pred_classes, df["class"].values):
        predicted_text = decode_class(predcition)
        print(f"{predicted_text}, {expected_text}")
        if predicted_text == expected_text:
            correct += 1
    accuracy = correct / total * 100
    print(f"\nAccuracy of classificator: {accuracy:.2f} %\n")

# inf no class column present, just print prediction
else:
    for predcition in pred_classes:
        print(decode_class(predcition))