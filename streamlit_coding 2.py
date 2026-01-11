# -*- coding: utf-8 -*-
"""Streamlit coding.ipynb"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Malaysia Air Pollution Index Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 700;
    color: #1f4e79;
    text-align: center;
}
.subtitle {
    font-size: 18px;
    text-align: center;
    color: #555;
}
.comment-box {
    background-color: #f4f6f9;
    padding: 12px;
    border-left: 5px solid #1f77b4;
    margin-top: 10px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🌫️ Air Pollution Index (API) Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Exploration • Modelling • Evaluation • Prediction</div>', unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("new_data.csv")

data = load_data()

# --------------------------------------------------
# PREPROCESSING & FEATURE ENGINEERING
# --------------------------------------------------
data = data.drop_duplicates()
data["air_pollution_concentration"] = data["air_pollution_concentration"].fillna(
    data["air_pollution_concentration"].median()
)

le = LabelEncoder()
data["air_pollutant_type"] = le.fit_transform(data["air_pollutant_type"])

limits = {
    "PM 2.5": 15,
    "PM 10": 45,
    "NO2": 0.02,
    "O3": 0.05,
    "CO": 4,
    "SO2": 0.02
}

pollutant_map = dict(zip(le.transform(le.classes_), le.classes_))

data["normalized_conc"] = data.apply(
    lambda row: row["air_pollution_concentration"] /
    limits[pollutant_map[row["air_pollutant_type"]]],
    axis=1
)

monthly_api = (
    data.groupby("month")["normalized_conc"]
    .mean()
    .reset_index()
    .rename(columns={"normalized_conc": "API_value"})
)

data = data.merge(monthly_api, on="month", how="left")

# Interaction features
data["traffic_emissions"] = data["car_registrations_y"] * data["normalized_conc"]
data["traffic_fire"] = data["car_registrations_y"] * data["fire_frp"]
data["ipi_pollution"] = data["ipi_index"] * data["normalized_conc"]
data["ipi_firefrp"] = data["ipi_index"] * data["fire_frp"]

features = [
    "avg_rainfall_mm",
    "fire_brightness",
    "fire_frp",
    "consumption",
    "normalized_conc",
    "traffic_emissions",
    "traffic_fire",
    "ipi_pollution",
    "ipi_firefrp"
]

X = data[features]
y = data["API_value"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==================================================
# TRAIN MODELS (CACHED)
# ==================================================
@st.cache_resource
def train_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

        results[name] = {
            "model": model,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
            "CV_R2": cv.mean(),
            "CV_R2_STD": cv.std()
        }
    return results

model_results = train_models()

# ==================================================
# SIDEBAR
# ==================================================
menu = st.sidebar.radio(
    "📌 Navigation",
    ["EDA", "Model Development", "Model Evaluation and Comparison", "Best Model & Prediction"]
)

# ==================================================
# EDA
# ==================================================
if menu == "EDA":
    st.subheader("📊 Exploratory Data Analysis")

    numeric_cols = data.select_dtypes(include=np.number).columns
    col = st.selectbox("Select feature", numeric_cols)

    fig, ax = plt.subplots()
    ax.hist(data[col], bins=30)
    ax.set_title(col)
    st.pyplot(fig)

    st.markdown("""
    <div class="comment-box">
    This distribution helps identify skewness, outliers, and overall variability 
    of the selected feature in the dataset.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(data[numeric_cols].describe())

# --------------------------------------------------
# MODEL DEVELOPMENT
# --------------------------------------------------
if menu == "Model Development":
    st.subheader("⚙️ Model Development")

    model_choice = st.selectbox(
        "Select Model",
        list(model_results.keys())
    )

    model = model_results[model_choice]["model"]
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    st.markdown("""
    <div class="comment-box">
    This plot compares actual vs predicted API values. 
    Points closer to the diagonal line indicate better model performance.
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# MODEL EVALUATION AND COMPARISON
# --------------------------------------------------
if menu == "Model Evaluation and Comparison":
    st.subheader("📈 Model Evaluation and Comparison")

    results_df = pd.DataFrame([
        {
            "Model": k,
            "MAE": v["MAE"],
            "RMSE": v["RMSE"],
            "R2": v["R2"],
            "CV R2 Mean": v["CV_R2"],
            "CV R2 STD": v["CV_R2_STD"]
        }
        for k, v in model_results.items()
    ])

    st.dataframe(results_df)

    metric = st.selectbox(
        "Select metric for comparison",
        ["R2", "RMSE", "MAE"]
    )

    fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df[metric])
    ax.set_title(f"Model Comparison based on {metric}")
    st.pyplot(fig)

# --------------------------------------------------
# BEST MODEL & PREDICTION
# --------------------------------------------------
if menu == "Best Model & Prediction":
    st.subheader("✅ Best Model: Gradient Boosting")

    st.markdown("""
    <div style="background-color:#eef3f8; padding:12px; border-left:5px solid #1f77b4;">
    Gradient Boosting is selected due to its strong predictive accuracy, ability to
    capture non-linear relationships, and stable generalisation performance.
    </div>
    """, unsafe_allow_html=True)

    gb_model = model_results["Gradient Boosting"]["model"]

    st.subheader("🔮 Predict API Index")
    st.caption("Adjust environmental and socio-economic factors to estimate air quality conditions.")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_rainfall = st.slider("🌧️ Average Rainfall (mm)",
                                 float(data["avg_rainfall_mm"].min()),
                                 float(data["avg_rainfall_mm"].max()),
                                 float(data["avg_rainfall_mm"].mean()))

        fire_brightness = st.slider("🔥 Fire Brightness",
                                    float(data["fire_brightness"].min()),
                                    float(data["fire_brightness"].max()),
                                    float(data["fire_brightness"].mean()))

        fire_frp = st.slider("🔥 Fire Radiative Power",
                             float(data["fire_frp"].min()),
                             float(data["fire_frp"].max()),
                             float(data["fire_frp"].mean()))

    with col2:
        consumption = st.slider("⚡ Electricity Consumption",
                                 float(data["consumption"].min()),
                                 float(data["consumption"].max()),
                                 float(data["consumption"].mean()))

        normalized_conc = st.slider("🫁 Normalised Pollution Concentration",
                                    float(data["normalized_conc"].min()),
                                    float(data["normalized_conc"].max()),
                                    float(data["normalized_conc"].mean()))

        traffic_emissions = st.slider("🚗 Traffic Emissions",
                                      float(data["traffic_emissions"].min()),
                                      float(data["traffic_emissions"].max()),
                                      float(data["traffic_emissions"].mean()))

    with col3:
        traffic_fire = st.slider("🚦 Traffic × Fire Interaction",
                                 float(data["traffic_fire"].min()),
                                 float(data["traffic_fire"].max()),
                                 float(data["traffic_fire"].mean()))

        ipi_pollution = st.slider("🏭 IPI × Pollution",
                                  float(data["ipi_pollution"].min()),
                                  float(data["ipi_pollution"].max()),
                                  float(data["ipi_pollution"].mean()))

        ipi_firefrp = st.slider("🏗️ IPI × Fire Radiative Power",
                                float(data["ipi_firefrp"].min()),
                                float(data["ipi_firefrp"].max()),
                                float(data["ipi_firefrp"].mean()))

    input_df = pd.DataFrame([[
        avg_rainfall,
        fire_brightness,
        fire_frp,
        consumption,
        normalized_conc,
        traffic_emissions,
        traffic_fire,
        ipi_pollution,
        ipi_firefrp
    ]], columns=features)

    prediction = gb_model.predict(scaler.transform(input_df))[0]

    if prediction <= 0.5:
        status, colour = "🟢 Good", "#2ecc71"
        advice = "Air quality is satisfactory with minimal health risks."
    elif prediction <= 1.0:
        status, colour = "🟡 Moderate", "#f1c40f"
        advice = "Acceptable air quality; sensitive groups should limit exposure."
    elif prediction <= 1.5:
        status, colour = "🟠 Unhealthy", "#e67e22"
        advice = "Increased health risks for vulnerable populations."
    else:
        status, colour = "🔴 Hazardous", "#e74c3c"
        advice = "Serious health effects expected; avoid outdoor activities."

    st.markdown(f"""
    <div style="background-color:{colour}; padding:20px; border-radius:12px; color:white;">
        <h2>Predicted API Value: {prediction:.4f}</h2>
        <h3>Air Quality Status: {status}</h3>
        <p>{advice}</p>
    </div>
    """, unsafe_allow_html=True)
