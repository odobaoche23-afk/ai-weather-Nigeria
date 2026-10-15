"""
Train humidity and wind-speed Random Forest models for 3, 5 and 7 days ahead,
and evaluate them (plus your existing rainfall/temperature models) on a
time-ordered hold-out set (the most recent 20% of the data).

Put this file in StreamlitProject/Models/ and run from StreamlitProject/:
    python Models/train_humidity_wind_models.py

Outputs:
    Models/Saved_Models/humidity_model_{3,5,7}_days.pkl
    Models/Saved_Models/wind_model_{3,5,7}_days.pkl
    Models/Saved_Models/metrics.json   (shown in the app's performance panel)

Assumptions (check them against your data):
  * weather_features.csv is DAILY data (one row per day)
  * It has Humidity and WindSpeed columns plus the 18 training features
  * The future value is the value N rows (days) later, per location if the
    file has a location column
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "Data" / "weather_features.csv"
SAVE = BASE / "Models" / "Saved_Models"
HORIZONS = [3, 5, 7]
TEST_FRACTION = 0.2

FEATURES = [
    "Rainfall", "MaxTemp", "MinTemp", "Humidity", "WindSpeed", "SolarRadiation",
    "Year", "Month", "Day", "DayOfYear",
    "Rainfall_Lag1", "Rainfall_Lag2", "Rainfall_Lag3",
    "MaxTemp_Lag1", "MaxTemp_Lag2", "MaxTemp_Lag3",
    "Rainfall_Avg7", "MaxTemp_Avg7",
]


def load_data():
    df = pd.read_csv(DATA)
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "time")), None)
    if date_col:
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif {"Year", "Month", "Day"}.issubset(df.columns):
        df["_date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
    else:
        raise ValueError("Cannot find a date column (or Year/Month/Day) to sort by time.")
    group = next((c for c in ["State", "Location", "City", "Station"] if c in df.columns), None)
    sort_cols = ([group] if group else []) + ["_date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df, group


def future(df, col, h, group):
    return df.groupby(group)[col].shift(-h) if group else df[col].shift(-h)


def score(y_true, y_pred, baseline, target, h):
    return {
        "target": target,
        "horizon_days": h,
        "test_rows": int(len(y_true)),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
        "Baseline_MAE": round(float(mean_absolute_error(y_true, baseline)), 3),
    }


def split(data):
    cut = int(len(data) * (1 - TEST_FRACTION))
    return data.iloc[:cut], data.iloc[cut:]


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    df, group = load_data()
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing}")
    results = []

    # ---- New models: humidity and wind speed ----
    new_targets = {
        "humidity": ("Humidity", "Humidity (%)"),
        "wind": ("WindSpeed", "Wind speed"),
    }
    for name, (col, label) in new_targets.items():
        for h in HORIZONS:
            data = df.copy()
            data["_y"] = future(data, col, h, group)
            data = data.dropna(subset=FEATURES + ["_y"]).sort_values("_date")
            train, test = split(data)

            model = RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42
            )
            model.fit(train[FEATURES], train["_y"])
            pred = model.predict(test[FEATURES])
            results.append(score(test["_y"], pred, test[col], label, h))
            print(f"{label} +{h}d  ->  {results[-1]}")

            # Refit on all data for the saved model
            model.fit(data[FEATURES], data["_y"])
            joblib.dump(model, SAVE / f"{name}_model_{h}_days.pkl")

    # ---- Existing models: evaluate only (not retrained) ----
    # NOTE: if these were trained on all rows or with a random split, the
    # hold-out numbers below are optimistic. Retrain with a time-ordered split
    # for an honest evaluation.
    existing = {
        "rainfall": ("Rainfall (mm)", "Rainfall", ["Rainfall_Future_{h}"]),
        "temperature": ("Temperature (°C)", "MaxTemp",
                        ["Temperature_Future_{h}", "MaxTemp_Future_{h}", "Temp_Future_{h}"]),
    }
    for name, (label, base_col, candidates) in existing.items():
        for h in HORIZONS:
            path = SAVE / f"{name}_model_{h}_days.pkl"
            target_col = next((c.format(h=h) for c in candidates if c.format(h=h) in df.columns), None)
            if not path.exists() or target_col is None:
                print(f"Skipping {label} +{h}d (model file or target column not found)")
                continue
            data = df.dropna(subset=FEATURES + [target_col]).sort_values("_date")
            _, test = split(data)
            model = joblib.load(path)
            pred = model.predict(test[FEATURES])
            if name == "rainfall":
                pred = np.clip(pred, 0, None)
            results.append(score(test[target_col], pred, test[base_col], label, h))
            print(f"{label} +{h}d  ->  {results[-1]}")

    with open(SAVE / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} metric rows to {SAVE / 'metrics.json'}")


if __name__ == "__main__":
    main()
