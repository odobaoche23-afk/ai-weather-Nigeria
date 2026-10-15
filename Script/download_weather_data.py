import os
import requests
import pandas as pd

# ============================================
# Folder where downloaded CSV files will be saved
# ============================================

DATA_FOLDER = "data"

os.makedirs(DATA_FOLDER, exist_ok=True)

# ============================================
# Nigerian State Capitals
# (Latitude, Longitude)
# ============================================

locations = {
    "Abuja": (9.0765, 7.3986),
    "Lagos": (6.5244, 3.3792),
    "Makurdi": (7.7337, 8.5364),
    "Kaduna": (10.5105, 7.4165),
    "Maiduguri": (11.8469, 13.1603),
    "Port_Harcourt": (4.8156, 7.0498),
    "Enugu": (6.4584, 7.5464)
}

# ============================================
# NASA POWER Parameters
# ============================================

parameters = [
    "PRECTOTCORR",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS2M"
]

parameter_string = ",".join(parameters)

# ============================================
# Download Function
# ============================================

def download_city(city, latitude, longitude):

    print(f"Downloading data for {city}...")

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={parameter_string}"
        "&community=RE"
        f"&longitude={longitude}"
        f"&latitude={latitude}"
        "&start=20180101"
        "&end=20251231"
        "&format=CSV"
    )

    response = requests.get(url)

    if response.status_code == 200:

        filename = os.path.join(DATA_FOLDER, f"{city}.csv")

        with open(filename, "wb") as file:
            file.write(response.content)

        print(f"Saved -> {filename}")

    else:

        print(f"Failed for {city}")


# ============================================
# Download Every City
# ============================================

for city, coordinates in locations.items():

    latitude = coordinates[0]
    longitude = coordinates[1]

    download_city(city, latitude, longitude)

print("\nFinished downloading all weather data.")