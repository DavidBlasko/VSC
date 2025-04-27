import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Function definition from task
def func(x):
    b = np.array([[0.2], [-1.31], [-0.9], [0.3]]).reshape(4, -1)
    r1 = np.sum((x - b)**2 - 10 * np.cos(0.7 * np.pi * (x + b)), axis=0)
    r2 = np.sum((x + b*2.1)**2 - 5 * np.cos(0.2 * np.pi * (x + b*4)), axis=0)
    return 40 + r1 + r2

# Dataset generation
np.random.seed(42)
N = 5000  # Num of samples
X = np.random.uniform(-7.4, 7.4, size=(N, 4)).astype(np.float32)
y = func(X.T).reshape(-1, 1).astype(np.float32)

# Transfer to PyTorch
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train
n_epochs = 300
for epoch in range(n_epochs):
    optimizer.zero_grad()
    outputs = net(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# Save model
torch.save(net.state_dict(), 'model.pth')
print("Model uložený do model.pth")
