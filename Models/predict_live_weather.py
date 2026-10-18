
import os
import joblib
import pandas as pd
from datetime import datetime, timedelta

from Models.live_weather import (
    get_live_weather,
    get_recent_daily_weather,
    get_daily_forecast
)


# =====================================================
# LOAD TRAINED MODELS
# =====================================================

MODEL_FOLDER = os.path.join("Models", "Saved_Models")

rainfall_models = {
    "3_days": joblib.load(
        os.path.join(MODEL_FOLDER, "rainfall_model_3_days.pkl")
    ),
    "5_days": joblib.load(
        os.path.join(MODEL_FOLDER, "rainfall_model_5_days.pkl")
    ),
    "7_days": joblib.load(
        os.path.join(MODEL_FOLDER, "rainfall_model_7_days.pkl")
    )
}

temperature_models = {
    "3_days": joblib.load(
        os.path.join(MODEL_FOLDER, "temperature_model_3_days.pkl")
    ),
    "5_days": joblib.load(
        os.path.join(MODEL_FOLDER, "temperature_model_5_days.pkl")
    ),
    "7_days": joblib.load(
        os.path.join(MODEL_FOLDER, "temperature_model_7_days.pkl")
    )
}


# =====================================================
# FEATURES USED DURING TRAINING
# =====================================================

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


# =====================================================
# WEATHER PREDICTION
# =====================================================

def predict_weather(state, forecast_date):

    # Use Nigeria's local calendar date.
    today = datetime.now().date()

    if isinstance(forecast_date, str):
        forecast_date = datetime.strptime(
            forecast_date, "%Y-%m-%d"
        )

    target_date = forecast_date.date()

    days_ahead = (target_date - today).days

    if days_ahead == 3:
        horizon = "3_days"
    elif days_ahead == 5:
        horizon = "5_days"
    elif days_ahead == 7:
        horizon = "7_days"
    else:
        raise ValueError(
            "Forecast date must be exactly 3, 5 or 7 days "
            "from today's date."
        )

    # -------------------------------------------------
    # Retrieve current conditions and recent daily data
    # -------------------------------------------------

        # Retrieve current conditions and recent daily data

    live = get_live_weather(state)
    external_forecast = get_daily_forecast(state, target_date)
    daily = get_recent_daily_weather(state)
    daily = daily.sort_values("Date").reset_index(drop=True)

    if len(daily) < 7:
        raise ValueError(
            "At least seven daily weather records are required."
        )

    # Use records up to today's date only.
    daily = daily[daily["Date"].dt.date <= today].reset_index(drop=True)

    if len(daily) < 7:
        raise ValueError(
            "The weather service did not provide enough recent "
            "daily records for this forecast."
        )

    # Latest available daily record and its preceding records.
    latest = daily.iloc[-1]
    previous_1 = daily.iloc[-2]
    previous_2 = daily.iloc[-3]
    previous_3 = daily.iloc[-4]

    recent_7 = daily.tail(7)

    # -------------------------------------------------
    # Build features using actual daily records
    # -------------------------------------------------

    features = {
        "Rainfall": float(latest["Rainfall"]),
        "MaxTemp": float(latest["MaxTemp"]),
        "MinTemp": float(latest["MinTemp"]),
        "Humidity": float(latest["Humidity"]),
        "WindSpeed": float(latest["WindSpeed"]),
        "SolarRadiation": float(latest["SolarRadiation"]),

        "Year": latest["Date"].year,
        "Month": latest["Date"].month,
        "Day": latest["Date"].day,
        "DayOfYear": latest["Date"].dayofyear,

        "Rainfall_Lag1": float(previous_1["Rainfall"]),
        "Rainfall_Lag2": float(previous_2["Rainfall"]),
        "Rainfall_Lag3": float(previous_3["Rainfall"]),

        "MaxTemp_Lag1": float(previous_1["MaxTemp"]),
        "MaxTemp_Lag2": float(previous_2["MaxTemp"]),
        "MaxTemp_Lag3": float(previous_3["MaxTemp"]),

        "Rainfall_Avg7": float(recent_7["Rainfall"].mean()),
        "MaxTemp_Avg7": float(recent_7["MaxTemp"].mean())
    }

    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    if X.isnull().any().any():
        raise ValueError(
            "Some required weather features are missing. "
            "Check the recent daily weather data."
        )

    # -------------------------------------------------
    # Generate predictions
    # -------------------------------------------------

    rainfall_prediction = float(
        rainfall_models[horizon].predict(X)[0]
    )

    temperature_prediction = float(
        temperature_models[horizon].predict(X)[0]
    )

    # Rainfall cannot physically be negative.
    rainfall_prediction = max(0.0, rainfall_prediction)

    return {
        "state": state,

        "generated_at": datetime.now().strftime(
            "%d %B %Y %H:%M"
        ),

        "forecast_date": target_date.strftime("%d %B %Y"),

        "forecast_horizon": horizon,

        "current_weather": {
            "rainfall": round(float(latest["Rainfall"]), 2),
            "temperature": round(float(live["Temperature"]), 2),
            "humidity": round(float(latest["Humidity"]), 2),
            "wind_speed": round(float(latest["WindSpeed"]), 2),
            "solar_radiation": round(
                float(latest["SolarRadiation"]), 2
            )
        },

               "prediction": {
            "rainfall": round(rainfall_prediction, 2),
            "temperature": round(temperature_prediction, 2)
        },

        "external_forecast": external_forecast    }


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    test_date = (
        datetime.now().date() + timedelta(days=3)
    ).strftime("%Y-%m-%d")

    result = predict_weather("Kaduna", test_date)

    print("\n" + "=" * 55)
    print("AI LIVE WEATHER FORECAST")
    print("=" * 55)

    print(f"State           : {result['state']}")
    print(f"Generated At    : {result['generated_at']}")
    print(f"Forecast Date   : {result['forecast_date']}")
    print(f"Forecast Model  : {result['forecast_horizon']}")

    print("\nRecent Daily Weather")
    print("-" * 55)
    print(f"Rainfall        : {result['current_weather']['rainfall']} mm/day")
    print(f"Temperature     : {result['current_weather']['temperature']} °C")
    print(f"Humidity        : {result['current_weather']['humidity']} %")
    print(f"Wind Speed      : {result['current_weather']['wind_speed']} m/s")
    print(
        "Solar Radiation : "
        f"{result['current_weather']['solar_radiation']} kWh/m²/day"
    )

    print("\nForecast for Selected Date")
    print("-" * 55)
    print(f"AI Rainfall        : {result['prediction']['rainfall']} mm")
    print(f"AI Max Temperature : {result['prediction']['temperature']} °C")

    print(
        f"Forecast Wind Speed: "
        f"{result['external_forecast']['wind_speed']} m/s"
    )
    print(
        f"Forecast Solar Radiation: "
        f"{result['external_forecast']['solar_radiation']} kWh/m²/day"
    )
