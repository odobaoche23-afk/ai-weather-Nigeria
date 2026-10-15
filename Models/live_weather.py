import requests

# ----------------------------------------
# Coordinates of Nigerian State Capitals
# ----------------------------------------

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
# ----------------------------------------
# Get Live Weather
# ----------------------------------------

def get_live_weather(state):

    if state not in STATE_COORDINATES:
        raise ValueError(f"Unknown state: {state}")

    latitude, longitude = STATE_COORDINATES[state]

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&current="
        "temperature_2m,"
        "relative_humidity_2m,"
        "wind_speed_10m,"
        "precipitation,"
        "shortwave_radiation"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()["current"]

    return {
        "State": state,
        "Temperature": float(data["temperature_2m"]),
        "Humidity": float(data["relative_humidity_2m"]),
        "WindSpeed": float(data["wind_speed_10m"]),
        "Rainfall": float(data["precipitation"]),
        "SolarRadiation": float(data["shortwave_radiation"])
    }


# ----------------------------------------
# Test
# ----------------------------------------

if __name__ == "__main__":

    weather = get_live_weather("Kaduna")

    print("\nCurrent Live Weather\n")

    for key, value in weather.items():
        print(f"{key}: {value}")