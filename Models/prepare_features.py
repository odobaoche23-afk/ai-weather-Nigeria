import os
import pandas as pd

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(BASE_DIR, "Data")

INPUT_FILE = os.path.join(
    DATA_FOLDER,
    "clean_weather_data.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_FOLDER,
    "weather_features.csv"
)

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading cleaned weather data...")

df = pd.read_csv(INPUT_FILE)

print("Data loaded successfully.")
print()

# ==========================================================
# CONVERT DATE
# ==========================================================

df["Date"] = pd.to_datetime(df["Date"])

# ==========================================================
# SORT DATA
# ==========================================================

df = df.sort_values(
    ["State", "Date"]
).reset_index(drop=True)

# ==========================================================
# DATE FEATURES
# ==========================================================

print("Creating calendar features...")

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfYear"] = df["Date"].dt.dayofyear
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

print("Done.")
print()

# ==========================================================
# WEATHER VARIABLES
# ==========================================================

weather_columns = [
    "Rainfall",
    "MaxTemp",
    "MinTemp",
    "Humidity",
    "WindSpeed",
    "SolarRadiation"
]

# ==========================================================
# CREATE LAG FEATURES
# ==========================================================

print("Creating lag features...")

for column in weather_columns:

    df[f"{column}_Lag1"] = (
        df.groupby("State")[column].shift(1)
    )

    df[f"{column}_Lag2"] = (
        df.groupby("State")[column].shift(2)
    )

    df[f"{column}_Lag3"] = (
        df.groupby("State")[column].shift(3)
    )

print("Lag features completed.")
print()

# ==========================================================
# CREATE ROLLING AVERAGES
# ==========================================================

print("Creating rolling averages...")

for column in weather_columns:

    df[f"{column}_Avg3"] = (
        df.groupby("State")[column]
        .transform(lambda x: x.rolling(3).mean())
    )

    df[f"{column}_Avg5"] = (
        df.groupby("State")[column]
        .transform(lambda x: x.rolling(5).mean())
    )

    df[f"{column}_Avg7"] = (
        df.groupby("State")[column]
        .transform(lambda x: x.rolling(7).mean())
    )

print("Rolling averages completed.")
print()

# ==========================================================
# CREATE AI TARGETS
# ==========================================================

print("Creating prediction targets...")

forecast_days = [3, 5, 7]

targets = [
    "Rainfall",
    "MaxTemp",
    "Humidity",
    "WindSpeed"
]

for target in targets:

    for day in forecast_days:

        df[f"{target}_Future_{day}"] = (
            df.groupby("State")[target]
            .shift(-day)
        )

print("Prediction targets completed.")
print()

# ==========================================================
# REMOVE EMPTY ROWS
# ==========================================================

print("Removing incomplete rows...")

df = df.dropna()

print()

print("Remaining rows:", len(df))

print()

# ==========================================================
# SAVE DATASET
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("========================================")
print("FEATURE ENGINEERING COMPLETE")
print("========================================")
print()

print("Saved to:")
print(OUTPUT_FILE)