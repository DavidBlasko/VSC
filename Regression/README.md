
# Regression

Modeling and approximating a complex function using an artificial neural network in Python.  
The server exposes an HTTP API for calculating predictions.

---

## 📚 Description

The task is to train a neural network that approximates a 4D function over the domain `[-7.4, 7.4]`, and then deploy it via an HTTP server.

The server provides three endpoints:
- `GET /` – documentation
- `POST /func` – prediction for multiple points
- `GET /func` – prediction for a single point via query parameters

---

## 🔧 Requirements

- Python 3.12
- Packages listed in `requirements.txt`

---

## ⚙️ Installation

1. Clone or download this repository.
2. Open a terminal in the project folder.
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   ```bash
   .\venv\Scripts\activate     # Windows
   source venv/bin/activate        # Linux/macOS
   ```
5. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Train the model

Run:
```bash
python model.py
```
The trained model will be saved to `model.pth`.

---

### 2. Start the server

Run:
```bash
python server.py
```
The server will be available at `http://localhost:5000`.

---

### 3. Test with the client

Run:
```bash
python client.py
```

---

## 🌐 API Endpoints

### GET /

- Returns: HTML documentation

---

### POST /func

- **Input:** JSON array of input points (each with 4 values).
- **Output:** JSON array of results.

#### Example request:
```json
[
  [2.647, 2.144, 0.112, 6.978],
  [0.4, -2.1, 0.07, -0.9]
]
```

#### Example response:
```json
[
  65.12345,
  44.98765
]
```

---

### GET /func?x1=..&x2=..&x3=..&x4=..

- **Input:** Query parameters `x1`, `x2`, `x3`, `x4`
- **Output:** Single prediction as plain text

#### Example:
```bash
GET http://localhost:5000/func?x1=0.1&x2=0.02&x3=-1.1&x4=0
```

#### Example response:
```
48.56789
```

---

## 📂 Project Structure

```
project2/
├── venv/              # Virtual environment (should be excluded from Git)
├── model.py           # Neural network training script
├── server.py          # HTTP server script
├── client.py          # Client script for testing the server
├── requirements.txt   # List of required packages
├── model.pth          # Saved trained model
└── README.md          # Project documentation
```

---

## 🛡️ Notes
- **Do not commit** the `venv/` folder to Git. Use a `.gitignore` file.
- The model file (`model.pth`) can be large; check project or course requirements regarding uploading it.

---

## 📜 License

This project is intended for educational purposes. ✨
