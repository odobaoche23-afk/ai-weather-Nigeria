import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("TEMPERATURE AI MODEL TRAINING")
print("=" * 60)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

DATA_PATH = os.path.join("Data", "weather_features.csv")

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows : {len(df)}")
print(f"Columns : {len(df.columns)}")

# -------------------------------------------------
# Use sample (recommended for 4GB RAM)
# -------------------------------------------------

df = df.sample(n=60000, random_state=42)

print(f"Training rows after sampling : {len(df)}")

# -------------------------------------------------
# Features
# -------------------------------------------------

FEATURE_COLUMNS = [

    "Rainfall",
    "MaxTemp",
    "MinTemp",
    "Humidity",
    "WindSpeed",
    "SolarRadiation",

    "Year",
    "Month",
    "Day",
    "DayOfYear",

    "Rainfall_Lag1",
    "Rainfall_Lag2",
    "Rainfall_Lag3",

    "MaxTemp_Lag1",
    "MaxTemp_Lag2",
    "MaxTemp_Lag3",

    "Rainfall_Avg7",
    "MaxTemp_Avg7"

]

# -------------------------------------------------
# Prediction Targets
# -------------------------------------------------

TARGETS = {

    "3_days": "MaxTemp_Future_3",
    "5_days": "MaxTemp_Future_5",
    "7_days": "MaxTemp_Future_7"

}

SAVE_FOLDER = os.path.join("Models", "Saved_Models")

os.makedirs(SAVE_FOLDER, exist_ok=True)

# -------------------------------------------------
# Train Models
# -------------------------------------------------

for model_name, target_column in TARGETS.items():

    print("\n" + "=" * 60)
    print(f"Training Temperature Model ({model_name})")
    print("=" * 60)

    X = df[FEATURE_COLUMNS]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")
    model.fit(X_train, y_train)
    print("Making predictions...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nModel Performance")
    print("-" * 30)
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.4f}")

    model_path = os.path.join(
        SAVE_FOLDER,
        f"temperature_model_{model_name}.pkl"
    )

    joblib.dump(model, model_path)

    print(f"Saved model -> {model_path}")

print("\n" + "=" * 60)
print("TEMPERATURE TRAINING COMPLETE")
print("=" * 60)