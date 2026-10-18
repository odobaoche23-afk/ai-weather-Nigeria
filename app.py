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
    initial_sidebar_state="expanded",
)

# =====================================================
# PROFESSIONAL VISUAL THEME (Times New Roman)
# =====================================================
st.markdown(
    """
    <style>
        :root {
            --navy: #16324F;
            --blue: #245B82;
            --teal: #168C8C;
            --ink: #243447;
            --muted: #526477;
            --line: #DCE4EC;
            --surface: #FFFFFF;
            --page: #F5F8FC;
        }

        /* Force Times New Roman across the whole app */
        html, body, [class*="css"], .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6,
        label, span, div, button, input, select, textarea {
            font-family: "Times New Roman", Times, serif !important;
        }

        .stApp {
            background: var(--page);
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: rgba(245, 248, 252, 0.96);
        }
        .block-container {
            max-width: 1440px;
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4,
        [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #16324F !important;
        }
        h1 {
            font-size: clamp(1.8rem, 3vw, 2.5rem) !important;
            font-weight: 750 !important;
        }
        h2, h3, h4 {
            font-weight: 700 !important;
        }
        p, li, label,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"],
        [data-testid="stExpander"] summary p {
            color: #243447;
        }
        [data-testid="stCaptionContainer"] {
            color: #526477 !important;
        }

        [data-testid="stSidebar"] {
            background: #EDF3F8;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #243447;
        }

        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 2px 8px rgba(22, 50, 79, 0.035);
        }
        [data-testid="stMetricLabel"] {
            color: #526477 !important;
            font-size: 0.88rem;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: #16324F !important;
            font-weight: 750;
        }

        .hero {
            background: linear-gradient(115deg, #16324F 0%, #245B82 70%, #168C8C 100%);
            padding: 30px 32px;
            border-radius: 18px;
            color: white;
            margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(22, 50, 79, 0.12);
        }
        .hero-kicker {
            color: #E0F0F7 !important;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .hero-title {
            color: #FFFFFF !important;
            font-size: clamp(1.65rem, 3vw, 2.35rem);
            font-weight: 750;
            line-height: 1.2;
            margin-bottom: 10px;
        }
        .hero-subtitle {
            color: #FFFFFF !important;
            font-size: 1rem;
            line-height: 1.6;
            max-width: 850px;
            margin-bottom: 0;
        }

        .section-intro {
            color: #526477 !important;
            margin-top: -8px;
            margin-bottom: 16px;
            font-size: 0.94rem;
            line-height: 1.55;
        }

        .panel {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 20px 22px;
            margin-bottom: 14px;
        }
        .panel-title {
            color: #16324F !important;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .panel-copy {
            color: #526477 !important;
            font-size: 0.92rem;
            line-height: 1.55;
            margin-bottom: 0;
        }
        .weather-symbol {
            font-size: 2rem;
            line-height: 1.3;
            margin-bottom: 8px;
        }
        .status-note {
            color: #34495E !important;
            font-size: 0.82rem;
            line-height: 1.6;
        }

        div.stButton > button,
        div.stButton > button[kind="primary"],
        div.stButton > button[data-testid="stBaseButton-primary"] {
            background: #16324F !important;
            color: #FFFFFF !important;
            border: 1px solid #16324F !important;
            border-radius: 9px;
            font-weight: 700 !important;
            min-height: 44px;
        }
        div.stButton > button p,
        div.stButton > button span {
            color: #FFFFFF !important;
        }
        div.stButton > button:hover {
            background: #245B82 !important;
            border-color: #245B82 !important;
            color: #FFFFFF !important;
        }

        div[data-testid="stExpander"] {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 10px;
        }
        div[data-testid="stExpander"] summary {
            color: #16324F !important;
            font-weight: 650;
        }

        [data-testid="stPlotlyChart"] {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 8px 8px 0 8px;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 10px;
        }

        .footer {
            border-top: 1px solid var(--line);
            margin-top: 32px;
            padding-top: 18px;
            text-align: center;
            color: #526477 !important;
            font-size: 0.82rem;
            line-height: 1.7;
        }
        .footer b, .footer i {
            color: #34495E !important;
        }

        @media (max-width: 700px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1rem;
            }
            .hero {
                padding: 23px 21px;
                border-radius: 13px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# CONSTANTS
# =====================================================
STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi",
    "Bayelsa", "Benue", "Borno", "Cross River", "Delta",
    "Ebonyi", "Edo", "Ekiti", "Enugu", "FCT", "Gombe",
    "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi",
    "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun",
    "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto",
    "Taraba", "Yobe", "Zamfara",
]
HORIZONS = (3, 5, 7)

RAIN_LOW = 2
RAIN_HEAVY = 10
RAIN_FLOOD = 20
TEMP_HOT = 32
TEMP_EXTREME = 35

# =====================================================
# HELPER FUNCTIONS
# =====================================================
@st.cache_data(ttl=1800, show_spinner=False)
def get_forecast(state: str, date_str: str):
    return predict_weather(state, date_str)

def rain_label(mm):
    if mm < RAIN_LOW:
        return "Very little rain"
    if mm < RAIN_HEAVY:
        return "Moderate rain"
    return "Heavy rain"

def temp_label(temp):
    if temp < 25:
        return "Cool"
    if temp < TEMP_HOT:
        return "Warm"
    return "Hot"

def weather_icon(rainfall, temperature):
    if rainfall >= RAIN_HEAVY:
        return "🌧️"
    if rainfall >= RAIN_LOW:
        return "🌦️"
    if temperature >= TEMP_HOT:
        return "☀️"
    if temperature < 25:
        return "☁️"
    return "🌤️"

def chart_layout(fig, y_title, x_title="Forecast horizon (days ahead)"):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Times New Roman, Times, serif", color="#243447", size=12),
        title=dict(font=dict(size=16, color="#16324F"), x=0.03, xanchor="left"),
        margin=dict(l=22, r=22, t=65, b=65),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color="#243447", size=13), standoff=14),
            showgrid=False,
            linecolor="#DCE4EC",
            tickfont=dict(color="#34495E"),
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color="#243447", size=13), standoff=10),
            gridcolor="#E9EEF4",
            zerolinecolor="#DCE4EC",
            linecolor="#DCE4EC",
            tickfont=dict(color="#34495E"),
            automargin=True,
        ),
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font_color="#16324F"),
    )
    return fig

def section_heading(title, description=None):
    st.subheader(title)
    if description:
        st.markdown(f'<p class="section-intro">{description}</p>', unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================
if "forecast_results" not in st.session_state:
    st.session_state.forecast_results = {}
if "forecast_errors" not in st.session_state:
    st.session_state.forecast_errors = {}
if "forecast_state" not in st.session_state:
    st.session_state.forecast_state = None
if "forecast_run_date" not in st.session_state:
    st.session_state.forecast_run_date = None

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">🌍 Weather intelligence · Nigeria</div>
        <div class="hero-title">AI Weather Prediction &amp; Climate Adaptation Tool</div>
        <p class="hero-subtitle">
            Short-term weather forecasts and practical climate adaptation
            guidance for farmers and local communities across Nigeria.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
with st.sidebar:
    st.markdown("## 🌦️ Forecast settings")
    st.caption("Choose a Nigerian state to generate a forecast.")

    selected_state = st.selectbox(
        "State or territory",
        STATES,
        index=STATES.index("FCT"),
    )

    st.markdown(
        """
        <div class="status-note">
            <b>Forecast horizons</b><br>
            3 days · 5 days · 7 days<br><br>
            The forecast uses the prediction models configured in this project.
        </div>
        """,
        unsafe_allow_html=True,
    )

    predict_button = st.button(
        "🌤️ Generate forecast",
        use_container_width=True,
        type="primary",
    )

# =====================================================
# RUN PREDICTION
# =====================================================
if predict_button:
    run_date = datetime.today().date()
    results = {}
    errors = {}

    with st.spinner(f"Generating forecasts for {selected_state}..."):
        for horizon in HORIZONS:
            target_date = run_date + timedelta(days=horizon)
            try:
                results[horizon] = get_forecast(
                    selected_state,
                    target_date.strftime("%Y-%m-%d"),
                )
            except Exception:
                errors[horizon] = traceback.format_exc()

    st.session_state.forecast_results = results
    st.session_state.forecast_errors = errors
    st.session_state.forecast_state = selected_state
    st.session_state.forecast_run_date = run_date.strftime("%Y-%m-%d")

# =====================================================
# FORECAST DISPLAY
# =====================================================
results = st.session_state.forecast_results
errors = st.session_state.forecast_errors

if results:
    first = results[min(results)]
    cw = first["current_weather"]

    # Forecast overview
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">📍 Forecast overview</div>
            <p class="panel-copy">
                Location: <b>{st.session_state.forecast_state}</b>
                &nbsp; · &nbsp;
                Generated: <b>{first["generated_at"]}</b>
                &nbsp; · &nbsp;
                Forecast horizons: <b>3, 5 and 7 days</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if errors:
        with st.expander("⚠️ Some forecasts could not be generated — view technical details"):
            for horizon, error_text in errors.items():
                st.markdown(f"**{horizon}-day forecast**")
                st.code(error_text)

    # Current weather
    section_heading(
        "🌍 Current weather conditions",
        "Values returned by the project's prediction module.",
    )

    current_rain = float(cw["rainfall"])
    current_temp = float(cw["temperature"])
    current_humidity = float(cw["humidity"])
    current_wind = float(cw["wind_speed"])
    current_solar = float(cw["solar_radiation"])
    current_icon = weather_icon(current_rain, current_temp)

    st.markdown(
        f'<div class="weather-symbol">{current_icon} {temp_label(current_temp)} conditions</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌧️ Rainfall", f"{current_rain:.2f} mm")
    c2.metric("🌡️ Temperature", f"{current_temp:.1f} °C")
    c3.metric("💧 Humidity", f"{current_humidity:.0f}%")
    c4.metric("💨 Wind speed", f"{current_wind:.2f} m/s")
    c5.metric("☀️ Solar radiation", f"{current_solar:.2f}")

    # Prepare forecast data
    rows = []
    for horizon, forecast in sorted(results.items()):
        raw_rainfall = float(forecast["prediction"]["rainfall"])
        temperature = float(forecast["prediction"]["temperature"])
        rainfall = max(raw_rainfall, 0.0)
        rows.append({
            "Horizon": f"{horizon} days",
            "Days ahead": horizon,
            "Date": forecast["forecast_date"],
            "Rainfall (mm)": rainfall,
            "Temperature (°C)": temperature,
        })
    fc = pd.DataFrame(rows)

    # AI forecast cards
    section_heading(
        "AI-generated forecast",
        "Compare predicted rainfall and temperature across the available forecast horizons.",
    )

    forecast_columns = st.columns(len(fc))
    for column, (_, row) in zip(forecast_columns, fc.iterrows()):
        rainfall = float(row["Rainfall (mm)"])
        temperature = float(row["Temperature (°C)"])
        symbol = weather_icon(rainfall, temperature)
        with column:
            st.markdown(
                f"""
                <div class="panel">
                    <div class="weather-symbol">{symbol}</div>
                    <div class="panel-title">{row["Horizon"]}</div>
                    <p class="panel-copy">{row["Date"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("🌧️ Predicted rainfall", f"{rainfall:.1f} mm")
            st.caption(rain_label(rainfall))
            st.metric("🌡️ Predicted temperature", f"{temperature:.1f} °C")
            st.caption(temp_label(temperature))

    # Charts
    section_heading("📊 Forecast charts")
    left, right = st.columns(2)

    with left:
        fig_rain = px.bar(
            fc, x="Horizon", y="Rainfall (mm)", text_auto=".1f",
            title="🌧️ Predicted rainfall",
            color_discrete_sequence=["#168C8C"],
            category_orders={"Horizon": ["3 days", "5 days", "7 days"]},
        )
        fig_rain.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
        fig_rain = chart_layout(fig_rain, "Rainfall (mm)")
        st.plotly_chart(fig_rain, use_container_width=True)

    with right:
        fig_temp = px.line(
            fc, x="Horizon", y="Temperature (°C)", markers=True,
            title="🌡️ Predicted temperature",
            color_discrete_sequence=["#245B82"],
            category_orders={"Horizon": ["3 days", "5 days", "7 days"]},
        )
        fig_temp.update_traces(line=dict(width=3), marker=dict(size=10, color="#168C8C"))
        fig_temp = chart_layout(fig_temp, "Temperature (°C)")
        st.plotly_chart(fig_temp, use_container_width=True)

    # Recommendation values
    max_rain = float(fc["Rainfall (mm)"].max())
    avg_rain = float(fc["Rainfall (mm)"].mean())
    max_temp = float(fc["Temperature (°C)"].max())
    peak_rain_day = fc.loc[fc["Rainfall (mm)"].idxmax(), "Horizon"]
    humidity = current_humidity
    wind = current_wind
    solar = current_solar

    # Weather summary
    section_heading("📝 Weather summary")
    with st.container(border=True):
        st.markdown("**Forecast summary**")
        for _, row in fc.iterrows():
            rainfall = float(row["Rainfall (mm)"])
            temperature = float(row["Temperature (°C)"])
            symbol = weather_icon(rainfall, temperature)
            st.write(
                f"{symbol} **{row['Horizon']} ({row['Date']}):** "
                f"{rain_label(rainfall).lower()} ({rainfall:.1f} mm), "
                f"{temp_label(temperature).lower()} ({temperature:.1f} °C)."
            )

    # Recommendations
    section_heading(
        "🌱 Climate adaptation recommendations",
        "Rule-based guidance derived from the forecast values. These are general suggestions, not official weather alerts.",
    )

    with st.expander("🚨 Disaster preparedness", expanded=True):
        if max_rain >= RAIN_FLOOD:
            st.error(
                f"Elevated rainfall-related flood concern: highest predicted rainfall is associated with the {peak_rain_day} horizon. "
                "Check local alerts and clear drainage channels where safe."
            )
            st.write("- Avoid flood-prone roads and locations if flooding occurs.")
            st.write("- Move harvested crops and essential supplies to safe storage.")
        elif max_rain >= RAIN_HEAVY:
            st.warning("Heavy rainfall is indicated by the model. Monitor local conditions and drainage systems.")
        else:
            st.success("The model's rainfall thresholds indicate a lower flood concern for this forecast period.")
        if max_temp >= TEMP_EXTREME:
            st.warning("High temperatures are predicted. Stay hydrated and limit strenuous outdoor work during the hottest part of the day.")

    with st.expander("🌾 Farming advice", expanded=True):
        if avg_rain < RAIN_LOW:
            st.write("- Consider irrigation where water is available and appropriate.")
            st.write("- Conserve available water and monitor soil moisture.")
        elif avg_rain < RAIN_HEAVY:
            st.write("- Review crop needs and local soil conditions when planning field activities.")
        else:
            st.write("- Consider delaying fertilizer application if heavy rain is imminent.")
            st.write("- Avoid spraying pesticides during rainfall or when label instructions advise against it.")
        if max_temp > TEMP_EXTREME:
            st.write("- Protect heat-sensitive crops where feasible.")
            st.write("- Check soil moisture and adjust irrigation to crop requirements.")

    with st.expander("☀️ Clean energy"):
        if solar > 500:
            st.write("- The returned solar-radiation value is relatively high under the project's existing thresholds.")
        elif solar > 250:
            st.write("- The returned solar-radiation value falls within the project's intermediate threshold range.")
        else:
            st.write("- The returned solar-radiation value is in the lower range under the project's existing thresholds.")
        if avg_rain >= RAIN_HEAVY:
            st.write("- During wet weather, keep batteries protected and plan for possible interruptions.")

    with st.expander("♻️ Waste management"):
        if max_rain >= RAIN_HEAVY:
            st.write("- Remove plastics and other waste from drainage channels where it is safe to do so.")
            st.write("- Secure refuse containers before rainfall.")
        else:
            st.write("- Maintain proper waste disposal and keep drainage paths clear.")

    with st.expander("🌳 Reforestation"):
        if avg_rain >= 5:
            st.write("- Forecast rainfall may be favourable for planting, subject to soil and local conditions.")
        else:
            st.write("- Water young trees as needed and check soil moisture.")
            st.write("- Time planting to local rainfall patterns and seedling requirements.")

    with st.expander("🔎 Technical forecast table"):
        st.dataframe(
            fc[["Horizon", "Date", "Rainfall (mm)", "Temperature (°C)"]],
            use_container_width=True,
            hide_index=True,
        )

elif errors:
    st.error("A forecast could not be generated. Open the technical details below to inspect the error.")
    with st.expander("Technical error details", expanded=True):
        for horizon, error_text in errors.items():
            st.markdown(f"**{horizon}-day forecast**")
            st.code(error_text)

else:
    # =====================================================
    # CLEAN HOME SCREEN
    # =====================================================
    section_heading(
        "Welcome",
        "Select a Nigerian state from the sidebar and click Generate forecast to view short-term weather predictions.",
    )

    with st.container(border=True):
        st.markdown("### What this tool provides")
        st.markdown(
            """
            - **Rainfall** estimates (mm)  
            - **Temperature** estimates (°C)  
            - **Humidity**, **wind speed** and **solar radiation** values  
            - Rule-based climate adaptation guidance for farmers and local communities  

            Forecasts are generated for **3 days**, **5 days** and **7 days** ahead.
            """
        )
        st.info("Choose a state in the sidebar, then click **Generate forecast**.")

# =====================================================
# FOOTER (kept exactly as requested)
# =====================================================
st.markdown(
    """
    <div class="footer">
        <b>Developed by Godwin Ochechema Adoyi</b><br>
        AI Weather Prediction &amp; Climate Adaptation Tool for Local
        Communities in Nigeria<br>
        Forecasts are model estimates. Cross-check important decisions
        with official sources such as NiMet.
    </div>
    """,
    unsafe_allow_html=True,
)