import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

# Load the trained XGBoost model
xgb_model = joblib.load('scripts\\best_xgb_model.pkl')

# Unique weather conditions from training data
weather_conditions = [
    'blowing dust', 'blowing dust / windy', 'blowing snow', 'blowing snow / windy', 'blowing snow nearby', 'clear', 
    'cloudy', 'cloudy / windy', 'drifting snow', 'drifting snow / windy', 'drizzle', 'drizzle / windy', 
    'drizzle and fog', 'dust whirls', 'duststorm', 'fair', 'fair / windy', 'fog', 'fog / windy', 
    'freezing drizzle', 'freezing rain', 'freezing rain / windy', 'funnel cloud', 'hail', 'haze', 
    'haze / windy', 'heavy blowing snow', 'heavy drizzle', 'heavy ice pellets', 'heavy rain', 
    'heavy rain / windy', 'heavy rain shower', 'heavy sleet', 'heavy sleet and thunder', 'heavy snow', 
    'heavy snow / windy', 'heavy snow with thunder', 'heavy t-storm', 'heavy t-storm / windy', 
    'heavy thunderstorms and rain', 'heavy thunderstorms and snow', 'ice pellets', 'light blowing snow', 
    'light drizzle', 'light drizzle / windy', 'light fog', 'light freezing drizzle', 'light freezing fog', 
    'light freezing rain', 'light freezing rain / windy', 'light haze', 'light ice pellets', 'light rain', 
    'light rain / windy', 'light rain shower', 'light rain showers', 'light rain with thunder', 'light sleet', 
    'light sleet / windy', 'light snow', 'light snow / windy', 'light snow and sleet', 'light snow and sleet / windy', 
    'light snow shower', 'light snow shower / windy', 'light snow with thunder', 'light thunderstorms and rain', 
    'light thunderstorms and snow', 'low drifting snow', 'mist', 'mist / windy', 'mostly cloudy', 
    'mostly cloudy / windy', 'n/a precipitation', 'overcast', 'partial fog', 'partly cloudy', 
    'partly cloudy / windy', 'patches of fog', 'patches of fog / windy', 'rain', 'rain / windy', 
    'rain shower', 'rain shower / windy', 'rain showers', 'sand / dust whirls nearby', 'sand / dust whirlwinds', 
    'scattered clouds', 'shallow fog', 'shallow fog / windy', 'showers in the vicinity', 'sleet', 'sleet / windy', 
    'sleet and thunder', 'small hail', 'smoke', 'smoke / windy', 'snow', 'snow / windy', 'snow and sleet', 
    'snow and sleet / windy', 'snow and thunder', 'snow grains', 'squalls', 'squalls / windy', 't-storm', 
    't-storm / windy', 'thunder', 'thunder / windy', 'thunder / wintry mix', 'thunder / wintry mix / windy', 
    'thunder and hail', 'thunder in the vicinity', 'thunderstorm', 'thunderstorms and rain', 'tornado', 
    'volcanic ash', 'widespread dust', 'widespread dust / windy', 'wintry mix', 'wintry mix / windy'
]

# Title and Introduction
st.title("Traffic Accident Severity Prediction")
st.markdown("""
This application predicts the severity of traffic accidents using the XGBoost model. Predictions are based on geospatial, weather, and road condition data.
""")

# Sidebar for user input
st.sidebar.header("Input Features")

# Function to get user input
def get_user_input():
    start_lat = st.sidebar.number_input("Start Latitude", value=34.0522, format="%f")
    start_lng = st.sidebar.number_input("Start Longitude", value=-118.2437, format="%f")
    distance = st.sidebar.number_input("Distance (mi)", value=0.5, format="%f")
    temperature = st.sidebar.number_input("Temperature (F)", value=70.0, format="%f")
    humidity = st.sidebar.number_input("Humidity (%)", value=50.0, format="%f")
    pressure = st.sidebar.number_input("Pressure (in)", value=29.92, format="%f")
    visibility = st.sidebar.number_input("Visibility (mi)", value=10.0, format="%f")
    wind_speed = st.sidebar.number_input("Wind Speed (mph)", value=5.0, format="%f")
    weather_condition = st.sidebar.selectbox("Weather Condition", weather_conditions)

    # Create DataFrame
    data = {
        "Start_Lat": [start_lat],
        "Start_Lng": [start_lng],
        "Distance(mi)": [distance],
        "Temperature(F)": [temperature],
        "Humidity(%)": [humidity],
        "Pressure(in)": [pressure],
        "Visibility(mi)": [visibility],
        "Wind_Speed(mph)": [wind_speed]
    }
    df = pd.DataFrame(data)

    # Add one-hot encoded columns for Weather Condition dynamically
    condition_columns = {f"Weather_Condition_{condition}": 0 for condition in weather_conditions}
    condition_columns[f"Weather_Condition_{weather_condition}"] = 1
    condition_df = pd.DataFrame([condition_columns])
    
    # Merge with main dataframe
    df = pd.concat([df, condition_df], axis=1)

    # Align with model features
    all_features = xgb_model.feature_names_in_
    for feature in all_features:
        if feature not in df.columns:
            df[feature] = 0  # Add missing columns as 0

    df = df[all_features]  # Ensure correct column order

    return df

# Collect user input
input_df = get_user_input()

# Display user input
st.subheader("User Input:")
st.write(input_df)

# Model Prediction
st.subheader("Prediction Results:")
xgb_prediction = xgb_model.predict(input_df)[0] + 1  # Adjust back to original labels

severity_map = {
    1: "Minor Impact",
    2: "Moderate Impact",
    3: "Severe Impact",
    4: "Critical Impact"
}

st.write(f"XGBoost Prediction: {severity_map[xgb_prediction]}")

# Map Visualization
st.subheader("Accident Location Map:")
m = folium.Map(location=[input_df["Start_Lat"].iloc[0], input_df["Start_Lng"].iloc[0]], zoom_start=13)
folium.Marker(
    [input_df["Start_Lat"].iloc[0], input_df["Start_Lng"].iloc[0]],
    popup=f"Predicted Severity: {severity_map[xgb_prediction]}"
).add_to(m)
st_folium(m, width=700, height=500)