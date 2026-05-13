import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_data(path):

    df = pd.read_csv(path)

    return df


def preprocess_data(df):

    # Remove unnecessary column
    df = df.drop("student_id", axis=1)

    # Features and target
    X = df.drop("total_score", axis=1)
    y = df["total_score"]

    # Convert grade column into numeric
    X = pd.get_dummies(X, columns=["grade"], drop_first=True)

    # Scale data
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    return X_train, X_test, y_train, y_test