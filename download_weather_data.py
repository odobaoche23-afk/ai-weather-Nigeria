import os
import time
import requests
import pandas as pd

# ====================================================
# PROJECT SETTINGS
# ====================================================

DATA_FOLDER = "Data"

CSV_FILE = os.path.join(DATA_FOLDER, "ng.csv")

START_DATE = "20100101"
END_DATE = "20251231"

PARAMETERS = [
    "PRECTOTCORR",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS2M",
    "ALLSKY_SFC_SW_DWN"
]

parameter_string = ",".join(PARAMETERS)

# ====================================================
# CHECK FILE EXISTS
# ====================================================

if not os.path.exists(CSV_FILE):
    print("ERROR:")
    print("Cannot find:")
    print(CSV_FILE)
    quit()

# ====================================================
# READ THE NIGERIA CITIES DATABASE
# ====================================================

df = pd.read_csv(CSV_FILE)

# ====================================================
# KEEP ONLY STATE CAPITALS + FCT
# ====================================================

capitals = df[
    (df["capital"] == "admin") |
    (df["capital"] == "primary")
].copy()

capitals = capitals.sort_values("admin_name")

print()
print("Found", len(capitals), "capital cities.")
print()

# ====================================================
# CREATE WEATHER FOLDER
# ====================================================

WEATHER_FOLDER = os.path.join(DATA_FOLDER, "Weather_Data")

os.makedirs(WEATHER_FOLDER, exist_ok=True)

# ====================================================
# DOWNLOAD LOOP
# ====================================================

total = len(capitals)

for index, row in enumerate(capitals.itertuples(), start=1):

    city = row.city
    state = row.admin_name
    latitude = row.lat
    longitude = row.lng

    print(f"[{index}/{total}] Downloading {state} ({city})...")

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters={parameter_string}"
        "&community=RE"
        f"&longitude={longitude}"
        f"&latitude={latitude}"
        f"&start={START_DATE}"
        f"&end={END_DATE}"
        "&format=CSV"
    )

    try:

        response = requests.get(url, timeout=60)

        if response.status_code == 200:

            filename = os.path.join(
                WEATHER_FOLDER,
                f"{state}.csv"
            )

            with open(filename, "wb") as file:
                file.write(response.content)

            print("   Saved")

        else:

            print("   Failed")

    except Exception as e:

        print("   Error:", e)

    time.sleep(1)

print()
print("===================================")
print("DOWNLOAD COMPLETE")
print("===================================")