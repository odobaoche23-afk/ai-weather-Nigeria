import os
import joblib
import pandas as pd
from datetime import datetime

from Models.live_weather import get_live_weather

# =====================================================
# LOAD TRAINED AI MODELS
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
# WEATHER PREDICTION FUNCTION
# =====================================================

def predict_weather(state, forecast_date):

    # -----------------------------------------
    # Get current live weather
    # -----------------------------------------

    live = get_live_weather(state)

    today = datetime.today()

    # Accept string dates such as "2026-08-05"
    if isinstance(forecast_date, str):
        forecast_date = datetime.strptime(
            forecast_date,
            "%Y-%m-%d"
        )

    days_ahead = (
        forecast_date.date() - today.date()
    ).days

    # Only forecasts that match trained models
    if days_ahead == 3:
        horizon = "3_days"

    elif days_ahead == 5:
        horizon = "5_days"

    elif days_ahead == 7:
        horizon = "7_days"

    else:
        raise ValueError(
            "Forecast date must be exactly 3, 5 or 7 days from today."
        )

    rainfall = live["Rainfall"]
    temperature = live["Temperature"]

    features = {

        "Rainfall": rainfall,
        "MaxTemp": temperature,
        "MinTemp": temperature - 5,

        "Humidity": live["Humidity"],
        "WindSpeed": live["WindSpeed"],
        "SolarRadiation": live["SolarRadiation"],

        "Year": today.year,
        "Month": today.month,
        "Day": today.day,
        "DayOfYear": today.timetuple().tm_yday,

        "Rainfall_Lag1": rainfall,
        "Rainfall_Lag2": rainfall,
        "Rainfall_Lag3": rainfall,

        "MaxTemp_Lag1": temperature,
        "MaxTemp_Lag2": temperature,
        "MaxTemp_Lag3": temperature,

        "Rainfall_Avg7": rainfall,
        "MaxTemp_Avg7": temperature

    }

    X = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS
    )
    # -----------------------------------------
    # Make AI Predictions
    # -----------------------------------------

    rainfall_prediction = round(
        float(
            rainfall_models[horizon].predict(X)[0]
        ),
        2
    )

    temperature_prediction = round(
        float(
            temperature_models[horizon].predict(X)[0]
        ),
        2
    )

    # -----------------------------------------
    # Return Results
    # -----------------------------------------

    return {

        "state": state,

        "generated_at": today.strftime(
            "%d %B %Y %H:%M"
        ),

        "forecast_date": forecast_date.strftime(
            "%d %B %Y"
        ),

        "forecast_horizon": horizon,

        "current_weather": {

            "rainfall": round(rainfall, 2),
            "temperature": round(temperature, 2),
            "humidity": round(live["Humidity"], 2),
            "wind_speed": round(live["WindSpeed"], 2),
            "solar_radiation": round(
                live["SolarRadiation"],
                2
            )

        },

        "prediction": {

            "rainfall": rainfall_prediction,
            "temperature": temperature_prediction

        }

    }
# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    # Change this date when testing.
    # It must be exactly 3, 5 or 7 days from today.
    test_date = "2026-08-04"

    result = predict_weather(
        "Kaduna",
        test_date
    )

    print("\n" + "=" * 55)
    print("AI LIVE WEATHER FORECAST")
    print("=" * 55)

    print(f"\nState           : {result['state']}")
    print(f"Generated At    : {result['generated_at']}")
    print(f"Forecast Date   : {result['forecast_date']}")
    print(f"Forecast Model  : {result['forecast_horizon']}")

    print("\nCurrent Weather")
    print("-" * 55)
    print(f"Rainfall        : {result['current_weather']['rainfall']} mm")
    print(f"Temperature     : {result['current_weather']['temperature']} °C")
    print(f"Humidity        : {result['current_weather']['humidity']} %")
    print(f"Wind Speed      : {result['current_weather']['wind_speed']} m/s")
    print(f"Solar Radiation : {result['current_weather']['solar_radiation']}")

    print("\nAI Forecast")
    print("-" * 55)
    print(f"Rainfall        : {result['prediction']['rainfall']} mm")
    print(f"Temperature     : {result['prediction']['temperature']} °C")
