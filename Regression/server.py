from flask import Flask, request, jsonify
import torch
import numpy as np

app = Flask(__name__)

# Definition of network - same as in model.py for training
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load model
net = Net()
net.load_state_dict(torch.load('model.pth'))
net.eval()

# Prediction function
def predict(x_array):
    x_tensor = torch.tensor(x_array, dtype=torch.float32)
    with torch.no_grad():
        y_pred = net(x_tensor).numpy()
    return y_pred.flatten()

@app.route("/", methods=["GET"])
def index():
    return """<h1>Documantation</h1>
    <p>POST /func: Send JSON with coordinates (4) and you get result.</p>
    <p>GET /func? x1=..&x2=..&x3=..&x4=..: Value of function.</p>
    """, 200

@app.route("/func", methods=["POST"])
def func_post():
    data = request.get_json()
    preds = predict(data)
    return jsonify(preds.tolist())

@app.route("/func", methods=["GET"])
def func_get():
    try:
        x1 = float(request.args.get('x1'))
        x2 = float(request.args.get('x2'))
        x3 = float(request.args.get('x3'))
        x4 = float(request.args.get('x4'))
        preds = predict([[x1, x2, x3, x4]])
        return str(preds[0])
    except Exception as e:
        return str(e), 400

if __name__ == "__main__":
    app.run(port=5000)
