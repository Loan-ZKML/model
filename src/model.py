import json
import numpy as np
import torch
import torch.nn as nn
from torch.onnx import export
import time
import sys
import os

# Get and validate command line arguments
if len(sys.argv) < 2:
    print("Usage: python3 model.py <train_data_path>")
    sys.exit(1)

train_data_path = sys.argv[1]
features = []
scores = []

try:
    with open(train_data_path, "r") as f:
      train_data = json.load(f)

    features = train_data['features']
    if not isinstance(features, list) or not all(isinstance(item, list) and len(item) == 4 for item in features):
        raise ValueError("Features must be a list of a list of 4 numbers")

    scores = train_data['scores']
    if not isinstance(scores, list) or not all(isinstance(item, float) for item in scores):
        raise ValueError("Scores must be a list of floats")
except json.JSONDecodeError:
    print("Error: Features must be a valid JSON array")
    sys.exit(1)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

# # Use output directory directly without creating an additional subdirectory
# os.makedirs(output_dir, exist_ok=True)

# Define a model that uses linear scaling instead of sigmoid
class LinearCreditScoreModel(nn.Module):
    def __init__(self):
        super(LinearCreditScoreModel, self).__init__()
        # Define weights for credit scoring features
        self.weights = nn.Parameter(torch.tensor([[0.25, 0.20, 0.25, 0.30]]).float())
        self.bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        # Linear combination of features
        raw_score = torch.matmul(x, self.weights.t()) + self.bias
        scaled_score = torch.clamp(raw_score, 0.0, 1.0)
        return scaled_score

# Convert features and scores to tensor
X_train = torch.tensor(features, dtype=torch.float32)
y_train = torch.tensor(scores, dtype=torch.float32)

# Create the model
model = LinearCreditScoreModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Export to ONNX
model_path = "credit_model.onnx"
dummy_input = X_train[0].unsqueeze(0)
export(
    model,
    dummy_input,
    model_path,
    verbose=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
