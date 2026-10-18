
import requests
import pandas as pd
from datetime import datetime

API_URL = "https://api.open-meteo.com/v1/forecast"


STATE_COORDINATES = {
    "Abia": (5.5320, 7.4860),
    "Adamawa": (9.2035, 12.4954),
    "Akwa Ibom": (5.0377, 7.9128),
    "Anambra": (6.2100, 7.0700),
    "Bauchi": (10.3158, 9.8442),
    "Bayelsa": (4.9267, 6.2676),
    "Benue": (7.7306, 8.5361),
    "Borno": (11.8333, 13.1500),
    "Cross River": (4.9589, 8.3269),
    "Delta": (6.2000, 6.7333),
    "Ebonyi": (6.3209, 8.1137),
    "Edo": (6.3382, 5.6258),
    "Ekiti": (7.6233, 5.2209),
    "Enugu": (6.4402, 7.4943),
    "FCT": (9.0765, 7.3986),
    "Federal Capital Territory": (9.0765, 7.3986),
    "Gombe": (10.2897, 11.1673),
    "Imo": (5.4850, 7.0350),
    "Jigawa": (11.7000, 9.3500),
    "Kaduna": (10.5105, 7.4165),
    "Kano": (12.0022, 8.5920),
    "Katsina": (12.9908, 7.6018),
    "Kebbi": (12.4539, 4.1975),
    "Kogi": (7.8000, 6.7333),
    "Kwara": (8.5000, 4.5500),
    "Lagos": (6.5244, 3.3792),
    "Nasarawa": (8.4833, 8.5167),
    "Niger": (9.6139, 6.5569),
    "Ogun": (7.1475, 3.3619),
    "Ondo": (7.2500, 5.2000),
    "Osun": (7.7667, 4.5667),
    "Oyo": (7.3775, 3.9470),
    "Plateau": (9.8965, 8.8583),
    "Rivers": (4.8156, 7.0498),
    "Sokoto": (13.0622, 5.2339),
    "Taraba": (8.8937, 11.3596),
    "Yobe": (11.7486, 11.9669),
    "Zamfara": (12.1704, 6.6641)
}


def get_coordinates(state):
    """Return the coordinates for a Nigerian state."""
    if state not in STATE_COORDINATES:
        raise ValueError(f"Unknown state: {state}")

    return STATE_COORDINATES[state]


def get_live_weather(state):
    """Retrieve current weather conditions for a state."""
    latitude, longitude = get_coordinates(state)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "precipitation,"
            "shortwave_radiation"
        ),
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "timezone": "Africa/Lagos"
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()

    current = response.json().get("current")

    if not current:
        raise ValueError("The weather API returned no current weather data.")

    return {
        "State": state,
        "Temperature": current["temperature_2m"],
        "Humidity": current["relative_humidity_2m"],
        "WindSpeed": current["wind_speed_10m"],
        "Rainfall": current["precipitation"],
        "SolarRadiation": current["shortwave_radiation"]
    }


def get_recent_daily_weather(state):
    """
    Retrieve recent daily weather observations and the current day's
    forecast where available.

    Rainfall: mm/day
    Temperature: degrees Celsius
    Humidity: percent
    Wind speed: m/s at 10 metres
    Solar radiation: kWh/m²/day
    """
    latitude, longitude = get_coordinates(state)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": (
            "precipitation_sum,"
            "temperature_2m_max,"
            "temperature_2m_min,"
            "relative_humidity_2m_mean,"
            "wind_speed_10m_mean,"
            "shortwave_radiation_sum"
        ),
        "past_days": 7,
        "forecast_days": 1,
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "timezone": "Africa/Lagos"
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()

    if "daily" not in payload:
        raise ValueError("The weather API did not return daily data.")

    daily_data = payload["daily"]

    # Open-Meteo reports daily shortwave radiation in MJ/m².
    # Convert to kWh/m² to match the usual NASA POWER unit.
    solar_values = daily_data["shortwave_radiation_sum"]

    daily = pd.DataFrame({
        "Date": daily_data["time"],
        "Rainfall": daily_data["precipitation_sum"],
        "MaxTemp": daily_data["temperature_2m_max"],
        "MinTemp": daily_data["temperature_2m_min"],
        "Humidity": daily_data["relative_humidity_2m_mean"],
        "WindSpeed": daily_data["wind_speed_10m_mean"],
        "SolarRadiation": [
            value / 3.6 if value is not None else None
            for value in solar_values
        ]
    })

    daily["Date"] = pd.to_datetime(daily["Date"])

    numeric_columns = [
        "Rainfall",
        "MaxTemp",
        "MinTemp",
        "Humidity",
        "WindSpeed",
        "SolarRadiation"
    ]

    daily[numeric_columns] = daily[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    daily = daily.dropna(subset=numeric_columns).reset_index(drop=True)
    daily["State"] = state

    if daily.empty:
        raise ValueError("No usable daily weather records were returned.")

    return daily


if __name__ == "__main__":
    print("Current weather:")
    print(get_live_weather("Kaduna"))

    print("\nRecent daily weather:")
    print(get_recent_daily_weather("Kaduna").to_string(index=False))

def get_daily_forecast(state, forecast_date):
    """Get external wind speed and solar radiation for a forecast date."""

    latitude, longitude = get_coordinates(state)

    if isinstance(forecast_date, (datetime, pd.Timestamp)):
        forecast_date = forecast_date.strftime("%Y-%m-%d")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "wind_speed_10m_mean,shortwave_radiation_sum",
        "start_date": forecast_date,
        "end_date": forecast_date,
        "wind_speed_unit": "ms",
        "timezone": "Africa/Lagos"
    }

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()

    daily = response.json().get("daily")

    if not daily or not daily.get("time"):
        raise ValueError(
            f"No external weather forecast returned for {forecast_date}."
        )

    wind_speed = daily["wind_speed_10m_mean"][0]
    solar_mj = daily["shortwave_radiation_sum"][0]

    return {
        "wind_speed": (
            round(float(wind_speed), 2)
            if wind_speed is not None else None
        ),
        "solar_radiation": (
            round(float(solar_mj) / 3.6, 2)
            if solar_mj is not None else None
        )
    }
