import requests
import json

# Test POST /func
x_data = [
    [2.647, 2.144, 0.112, 6.978],
    [0.4, -2.1, 0.07, -0.9]
]

response = requests.post('http://localhost:5000/func', json=x_data)
if response.status_code == 200:
    print("POST /func Result:", response.json())
else:
    print("POST /func Error:", response.status_code, response.text)

# Test GET /func
params = {
    'x1': 0.1,
    'x2': 0.02,
    'x3': -1.1,
    'x4': 0.0
}

response = requests.get('http://localhost:5000/func', params=params)
if response.status_code == 200:
    print("GET /func Result:", response.text)
else:
    print("GET /func Error:", response.status_code, response.text)
