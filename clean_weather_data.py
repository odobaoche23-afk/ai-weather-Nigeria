import os
import pandas as pd

# ==========================================
# PROJECT FOLDERS
# ==========================================

DATA_FOLDER = "Data"
WEATHER_FOLDER = os.path.join(DATA_FOLDER, "Weather_Data")

OUTPUT_FILE = os.path.join(DATA_FOLDER, "clean_weather_data.csv")

# ==========================================
# STORE ALL STATES
# ==========================================

all_data = []

# ==========================================
# READ EVERY CSV
# ==========================================

for filename in os.listdir(WEATHER_FOLDER):

    if not filename.endswith(".csv"):
        continue

    filepath = os.path.join(WEATHER_FOLDER, filename)

    state = filename.replace(".csv", "")

    print(f"Reading {state}...")

    # --------------------------------------
    # Find where the real data starts
    # --------------------------------------

    start_line = 0

    with open(filepath, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file):

            if line.startswith("YEAR"):

                start_line = line_number
                break

    # --------------------------------------
    # Read only the table
    # --------------------------------------

    df = pd.read_csv(filepath, skiprows=start_line)

    # --------------------------------------
    # Add State Column
    # --------------------------------------

    df["State"] = state

    # --------------------------------------
    # Create Date Column
    # --------------------------------------

    df["Date"] = pd.to_datetime(
        dict(
            year=df["YEAR"],
            month=df["MO"],
            day=df["DY"]
        )
    )

    # --------------------------------------
    # Keep Required Columns
    # --------------------------------------

    df = df[
        [
            "State",
            "Date",
            "PRECTOTCORR",
            "T2M_MAX",
            "T2M_MIN",
            "RH2M",
            "WS2M",
            "ALLSKY_SFC_SW_DWN"
        ]
    ]

    # --------------------------------------
    # Rename Columns
    # --------------------------------------

    df.columns = [
        "State",
        "Date",
        "Rainfall",
        "MaxTemp",
        "MinTemp",
        "Humidity",
        "WindSpeed",
        "SolarRadiation"
    ]

    # --------------------------------------
    # Remove Missing Values
    # --------------------------------------

    df = df.replace(-999, pd.NA)

    df = df.dropna()

    all_data.append(df)

# ==========================================
# MERGE ALL STATES
# ==========================================

weather = pd.concat(all_data, ignore_index=True)

# ==========================================
# SORT
# ==========================================

weather = weather.sort_values(
    ["State", "Date"]
)

# ==========================================
# SAVE
# ==========================================

weather.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("===================================")
print("Cleaning Complete")
print("===================================")

print()

print("Rows :", len(weather))

print("Columns :", len(weather.columns))

print()

print("Saved to:")

print(OUTPUT_FILE)