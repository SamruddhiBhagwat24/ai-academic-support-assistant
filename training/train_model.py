import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.preprocessing import load_data, preprocess_data

# Load dataset
df = load_data("data/student_data.csv")

# Preprocess dataset
X_train, X_test, y_train, y_test = preprocess_data(df)


# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# Neural Network Model
class StudentPerformanceModel(nn.Module):

    def __init__(self, input_size):

        super(StudentPerformanceModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x


# Dynamic input size
input_size = X_train.shape[1]

model = StudentPerformanceModel(input_size)


# Loss and optimizer
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
epochs = 100

for epoch in range(epochs):

    model.train()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


# Evaluation
model.eval()

with torch.no_grad():

    predictions = model(X_test)

    test_loss = criterion(predictions, y_test)

    print(f"\nTest Loss: {test_loss.item():.4f}")


# Save model
torch.save(model.state_dict(), "models/student_model.pth")

print("\nModel saved successfully.")