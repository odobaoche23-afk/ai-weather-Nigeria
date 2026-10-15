import os
import joblib
import pandas as pd

# =====================================================
# Load Weather Dataset
# =====================================================

DATA_PATH = os.path.join("Data", "weather_features.csv")

weather_data = pd.read_csv(DATA_PATH)

weather_data["Date"] = pd.to_datetime(weather_data["Date"])

# =====================================================
# Load Trained Models
# =====================================================

MODEL_FOLDER = os.path.join("Models", "Saved_Models")

rainfall_models = {

    3: joblib.load(os.path.join(MODEL_FOLDER, "rainfall_model_3_days.pkl")),
    5: joblib.load(os.path.join(MODEL_FOLDER, "rainfall_model_5_days.pkl")),
    7: joblib.load(os.path.join(MODEL_FOLDER, "rainfall_model_7_days.pkl"))

}

temperature_models = {

    3: joblib.load(os.path.join(MODEL_FOLDER, "temperature_model_3_days.pkl")),
    5: joblib.load(os.path.join(MODEL_FOLDER, "temperature_model_5_days.pkl")),
    7: joblib.load(os.path.join(MODEL_FOLDER, "temperature_model_7_days.pkl"))

}

# =====================================================
# Features Used By AI
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
# Weather Prediction Function
# =====================================================

def predict_weather(state_name):

    state = weather_data[
        weather_data["State"] == state_name
    ].copy()

    if len(state) == 0:
        raise ValueError("State not found.")

    state = state.sort_values("Date")

    latest = state.iloc[-1]
    # ---------------------------------------------
    # Create feature row for prediction
    # ---------------------------------------------

    feature_row = pd.DataFrame(
        [[latest[col] for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    rainfall_forecast = {}
    temperature_forecast = {}

    # ---------------------------------------------
    # Rainfall Forecast
    # ---------------------------------------------

    for days, model in rainfall_models.items():

        prediction = float(model.predict(feature_row)[0])

        if prediction < 0:
            prediction = 0.0

        rainfall_forecast[f"{days}_days"] = round(prediction, 2)

    # ---------------------------------------------
    # Maximum Temperature Forecast
    # ---------------------------------------------

    for days, model in temperature_models.items():

        prediction = float(model.predict(feature_row)[0])

        temperature_forecast[f"{days}_days"] = round(prediction, 2)

    # ---------------------------------------------
    # Return prediction results
    # ---------------------------------------------

    return {

        "state": state_name,

        "forecast_based_on": latest["Date"].strftime("%d %B %Y"),

        "rainfall": rainfall_forecast,

        "temperature": temperature_forecast

    }


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    result = predict_weather("Benue")

    print("\nForecast Results")
    print(result)