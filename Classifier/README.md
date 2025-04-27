# Binary Classifier (Neural Network)

This project implements a simple **binary classifier** using a **neural network** in **Python 3.12** and **TensorFlow**.

It consists of two scripts:

- **`train.py`** – trains the model on a CSV dataset and saves the trained model.
- **`inference.py`** – uses the trained model to predict classes for new data and evaluates accuracy if ground truth labels are available.

---

## 🧩 Requirements

- Python 3.8–3.12
- Virtual environment (recommended)

Install dependencies:
```bash
pip install -r requirements.txt
```

Required Python packages:
- pandas
- numpy
- scikit-learn
- tensorflow

---

## 🛠️ How to Use

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate        # On Windows
source .venv/bin/activate       # On Linux/macOS
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

---

### 3. Train the model

Train the classifier with your training data (CSV file with features and `class` label):

```bash
python train.py path/to/p1_train.csv
```

- This saves:
  - The trained model (`model.h5`)
  - The feature scaler parameters (`scaler.npz`)

---

### 4. Run inference

Predict classes for new data:

```bash
python inference.py path/to/test_data.csv
```

- If the test CSV includes a `class` column:
  - The script outputs the prediction and the expected label, and computes the overall accuracy.
  
- If the test CSV does **not** include a `class` column:
  - The script outputs only the predicted classes.

Example output:
```
QqQqQ, QqQqQ
MmMmM, QqQqQ
QqQqQ, MmMmM
...
Classifier accuracy: 89.25 %
```
or
```
QqQqQ
MmMmM
QqQqQ
...
```

---

## 📄 Notes

- The input CSV must have 31 feature columns (`f00` to `f30`) and optionally a `class` column (`QqQqQ` or `MmMmM`).
- The model expects features scaled with the saved scaler.
- Model outputs binary classes: **`QqQqQ`** (0) or **`MmMmM`** (1).

---

## 📌 Project Structure

```
├── train.py          # Training script
├── inference.py      # Inference script
├── requirements.txt  # Python package dependencies
├── model.h5          # Saved trained model (after training)
├── scaler.npz        # Saved scaler parameters (after training)
├── p1_train.csv      # Training dataset
├── p1_test_student.csv   # Test dataset
└── README.md         # Project documentation
```