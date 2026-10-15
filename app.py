import traceback
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from Models.predict_live_weather import predict_weather

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="AI Weather Prediction & Climate Adaptation Tool",
    page_icon="🌤️",
    layout="wide",
)

st.title("🌤 AI Weather Prediction & Climate Adaptation Tool")
st.caption("Supporting Local Farmers and Communities in Nigeria")
st.markdown("---")

# =====================================================
# CONSTANTS
# =====================================================
STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue",
    "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "FCT",
    "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi",
    "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo",
    "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara",
]
HORIZONS = (3, 5, 7)

# Rainfall thresholds (mm) - one place, used by the summary and recommendations
RAIN_LOW = 2        # below this: very little rain
RAIN_HEAVY = 10     # at or above: heavy rain
RAIN_FLOOD = 20     # at or above: high flood risk
TEMP_HOT = 32
TEMP_EXTREME = 35


# =====================================================
# HELPERS
# =====================================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_forecast(state: str, date_str: str):
    """Cache each (state, date) call for 30 minutes so reruns don't refetch live data."""
    return predict_weather(state, date_str)


def rain_label(mm):
    if mm < RAIN_LOW:
        return "Very little rain"
    if mm < RAIN_HEAVY:
        return "Moderate rain"
    return "Heavy rain"


def temp_label(t):
    if t < 25:
        return "Cool"
    if t < TEMP_HOT:
        return "Warm"
    return "Hot"


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Weather Forecast")

selected_state = st.sidebar.selectbox(
    "Select State",
    STATES,
    index=STATES.index("FCT"),  # change to "Benue" etc. if you prefer another default
)

st.sidebar.caption("The tool forecasts 3, 5 and 7 days ahead together.")
predict_button = st.sidebar.button("Predict Weather", use_container_width=True)

# =====================================================
# RUN AI PREDICTION
# =====================================================
if predict_button:
    today = datetime.today().date()
    results, errors = {}, {}

    with st.spinner("Generating AI weather forecast..."):
        for h in HORIZONS:
            target_date = today + timedelta(days=h)
            try:
                results[h] = get_forecast(selected_state, target_date.strftime("%Y-%m-%d"))
            except Exception:
                errors[h] = traceback.format_exc()

    if errors:
        with st.expander("⚠️ Some forecasts failed (click for details)", expanded=not results):
            for h, tb in errors.items():
                st.write(f"**{h}-day forecast**")
                st.code(tb)
    if not results:
        st.error("No forecast could be generated. See the error details above.")
        st.stop()

    first = results[min(results)]
    st.success("Forecast generated successfully.")

    # ---------------- Forecast information ----------------
    st.markdown("---")
    st.subheader("📍 Forecast Information")
    i1, i2 = st.columns(2)
    i1.metric("State", first["state"])
    i2.metric("Generated", first["generated_at"])

    # ---------------- Current live weather ----------------
    cw = first["current_weather"]
    st.markdown("---")
    st.subheader("🌤 Current Live Weather")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌧 Rainfall", f"{cw['rainfall']:.2f} mm")
    c2.metric("🌡 Temperature", f"{cw['temperature']:.2f} °C")
    c3.metric("💧 Humidity", f"{cw['humidity']:.0f}%")
    c4.metric("💨 Wind Speed", f"{cw['wind_speed']:.2f} m/s")
    c5.metric("☀ Solar Radiation", f"{cw['solar_radiation']:.0f}")

    # ---------------- AI forecast (all horizons) ----------------
    st.markdown("---")
    st.subheader("🔮 AI Forecast")

    rows = []
    for h, r in sorted(results.items()):
        rows.append(
            {
                "Horizon": f"{h} days",
                "Date": r["forecast_date"],
                "Rainfall (mm)": max(float(r["prediction"]["rainfall"]), 0.0),
                "Temperature (°C)": float(r["prediction"]["temperature"]),
            }
        )
    fc = pd.DataFrame(rows)

    cols = st.columns(len(fc))
    for col, (_, row) in zip(cols, fc.iterrows()):
        with col:
            st.markdown(f"**{row['Horizon']}** · {row['Date']}")
            st.metric("🌧 Rainfall", f"{row['Rainfall (mm)']:.2f} mm", rain_label(row["Rainfall (mm)"]), delta_color="off")
            st.metric("🌡 Temperature", f"{row['Temperature (°C)']:.2f} °C", temp_label(row["Temperature (°C)"]), delta_color="off")

    left, right = st.columns(2)
    with left:
        fig_r = px.bar(fc, x="Horizon", y="Rainfall (mm)", title="Predicted Rainfall", text_auto=".1f")
        st.plotly_chart(fig_r, use_container_width=True)
    with right:
        fig_t = px.line(fc, x="Horizon", y="Temperature (°C)", markers=True, title="Predicted Temperature")
        st.plotly_chart(fig_t, use_container_width=True)

    # Values used for the rules
    max_rain = fc["Rainfall (mm)"].max()
    avg_rain = fc["Rainfall (mm)"].mean()
    max_temp = fc["Temperature (°C)"].max()
    peak_rain_day = fc.loc[fc["Rainfall (mm)"].idxmax(), "Horizon"]
    humidity = cw["humidity"]
    wind = cw["wind_speed"]
    solar = cw["solar_radiation"]

    # ---------------- Summary ----------------
    st.markdown("---")
    st.subheader("📝 Weather Summary")
    st.markdown("**Forecast (AI model)**")
    for _, row in fc.iterrows():
        st.write(f"• {row['Horizon']}: {rain_label(row['Rainfall (mm)']).lower()} "
                 f"({row['Rainfall (mm)']:.1f} mm), "
                 f"{temp_label(row['Temperature (°C)']).lower()} ({row['Temperature (°C)']:.1f} °C).")

    st.markdown("**Current conditions (live data)**")
    st.write("• " + ("High humidity." if humidity >= 80 else "Moderate humidity." if humidity >= 60 else "Low humidity."))
    st.write("• " + ("Light winds." if wind < 3 else "Moderate winds." if wind < 8 else "Strong winds."))
    st.write("• " + ("Strong sunshine." if solar > 500 else "Moderate sunshine." if solar > 200 else "Cloudy conditions."))

    # ---------------- Recommendations ----------------
    st.markdown("---")
    st.subheader("🌱 Climate Adaptation Recommendations")
    st.caption("Based on the 3, 5 and 7-day forecasts and current conditions.")

    with st.expander("🚨 Disaster Preparedness", expanded=True):
        if max_rain >= RAIN_FLOOD:
            st.error(f"High flood risk (peak rainfall at the {peak_rain_day} mark). Clear drainage channels immediately.")
            st.write("• Avoid flood-prone roads and communities.")
            st.write("• Move harvested crops to safe storage.")
        elif max_rain >= RAIN_HEAVY:
            st.warning("Heavy rainfall expected. Monitor local drainage systems.")
        else:
            st.success("Low flood risk.")
        if max_temp >= TEMP_EXTREME:
            st.warning("Heat stress risk. Stay hydrated and avoid outdoor work at midday.")

    with st.expander("🌾 Farming Advice", expanded=True):
        if avg_rain < RAIN_LOW:
            st.write("• Irrigation is strongly recommended.")
            st.write("• Conserve available water.")
        elif avg_rain < RAIN_HEAVY:
            st.write("• Weather is suitable for most crops.")
        else:
            st.write("• Delay fertilizer application until rainfall reduces.")
            st.write("• Avoid pesticide spraying during heavy rain.")
        if max_temp > TEMP_EXTREME:
            st.write("• Protect crops from excessive heat.")
            st.write("• Increase irrigation frequency.")

    with st.expander("☀ Clean Energy", expanded=True):
        if solar > 500:
            st.write("• Excellent conditions for solar energy generation.")
        elif solar > 250:
            st.write("• Good conditions for solar energy.")
        else:
            st.write("• Solar output may be reduced because of cloud cover.")
        if avg_rain >= RAIN_HEAVY:
            st.write("• Rainy days ahead: keep batteries charged and plan backup supply.")

    with st.expander("♻ Waste Management", expanded=True):
        if max_rain >= RAIN_HEAVY:
            st.write("• Clear plastics from drainage channels.")
            st.write("• Secure refuse bins before rainfall.")
        else:
            st.write("• Maintain proper waste disposal.")

    with st.expander("🌳 Reforestation", expanded=True):
        if avg_rain >= 5:
            st.write("• Good conditions for tree planting.")
        else:
            st.write("• Water young trees regularly.")
            st.write("• Delay planting new seedlings if possible.")

    with st.expander("🔎 Raw forecast data"):
        st.dataframe(fc, use_container_width=True, hide_index=True)

# =====================================================
# DEFAULT HOME SCREEN
# =====================================================
else:
    st.info("Select a Nigerian state, then click **Predict Weather** to see the 3, 5 and 7-day forecasts.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("AI-Based Weather Prediction & Climate Adaptation Tool for Local Communities in Nigeria")
st.caption("Forecasts are model estimates. Always cross-check with official meteorological alerts (e.g. NiMet).")
