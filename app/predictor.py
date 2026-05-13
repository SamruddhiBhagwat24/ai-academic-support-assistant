import torch
import torch.nn as nn
import joblib
import pandas as pd


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


# Load scaler
scaler = joblib.load("models/scaler.pkl")


# Input feature size
input_size = scaler.n_features_in_

# Load model
model = StudentPerformanceModel(input_size)

model.load_state_dict(
    torch.load("models/student_model.pth")
)

model.eval()


def predict_score(
    weekly_self_study_hours,
    attendance_percentage,
    class_participation,
    grade
):

    # Create dataframe
    input_data = pd.DataFrame([{
        "weekly_self_study_hours": weekly_self_study_hours,
        "attendance_percentage": attendance_percentage,
        "class_participation": class_participation,
        "grade": grade
    }])

    # One-hot encoding
    input_data = pd.get_dummies(
        input_data,
        columns=["grade"]
    )

    # Match training columns
    expected_columns = scaler.feature_names_in_

    for col in expected_columns:

        if col not in input_data.columns:

            input_data[col] = 0

    input_data = input_data[expected_columns]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Convert to tensor
    input_tensor = torch.tensor(
        input_scaled,
        dtype=torch.float32
    )

    # Prediction
    with torch.no_grad():

        prediction = model(input_tensor)

    return prediction.item()