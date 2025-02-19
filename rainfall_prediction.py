import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    with open('rainfall_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def encode_wind_direction(direction):
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return directions.index(direction) if direction in directions else -1

# Predict rainfall
def rainfall_prediction(input_data):
    model = load_model()
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction

def main():
    st.set_page_config(page_title="Rainfall Prediction", page_icon="☔", layout="wide")
    
    # Custom CSS for styling
    st.markdown("""
        <style>
            .stButton>button {background-color: #4CAF50; color: white; padding: 10px 24px;}
            .stButton>button:hover {background-color: #45a049;}
            .stNumberInput>div>div>input, .stSelectbox>div>div>select {font-size: 16px;}
            .created-by {font-size: 20px; font-weight: bold; color: #4CAF50;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("☔ Rainfall Prediction")
    st.markdown("This app predicts the likelihood of rainfall based on various weather parameters.")

    with st.sidebar:
        st.markdown('<p class="created-by">Created by Andrew O.A.</p>', unsafe_allow_html=True)
        try:
            profile_pic = Image.open("prof.jpeg")
            st.image(profile_pic, caption="Andrew O.A.", use_container_width=True, output_format="JPEG")
        except:
            st.warning("Profile image not found.")
        st.info("This app uses a machine learning model trained on weather data to predict rainfall.")
        st.markdown("[GitHub](https://github.com/Andrew-oduola) | [LinkedIn](https://linkedin.com/in/andrew-oduola-django-developer)")

    
    result_placeholder = st.empty()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        MinTemp = st.number_input("Min Temperature (°C)", min_value=-10.0, max_value=50.0, value=15.0)
        MaxTemp = st.number_input("Max Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
        Rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=1.0)
        Evaporation = st.number_input("Evaporation (mm)", min_value=0.0, value=5.0)
        Sunshine = st.number_input("Sunshine (hours)", min_value=0.0, value=8.0)

    with col2:
        WindGustDir = st.selectbox("Wind Gust Direction", ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
        WindGustSpeed = st.number_input("Wind Gust Speed (km/h)", min_value=0, value=30)
        WindDir9am = st.selectbox("Wind Direction at 9 AM", ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
        WindDir3pm = st.selectbox("Wind Direction at 3 PM", ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
        WindSpeed9am = st.number_input("Wind Speed at 9 AM (km/h)", min_value=0, value=15)

    
    with col3:
        WindSpeed3pm = st.number_input("Wind Speed at 3 PM (km/h)", min_value=0, value=20)
        Humidity9am = st.number_input("Humidity at 9 AM (%)", min_value=0, max_value=100, value=60)
        Humidity3pm = st.number_input("Humidity at 3 PM (%)", min_value=0, max_value=100, value=50)
        Pressure9am = st.number_input("Pressure at 9 AM (hPa)", min_value=900, max_value=1100, value=1013)
        Pressure3pm = st.number_input("Pressure at 3 PM (hPa)", min_value=900, max_value=1100, value=1010)

    with col4:
        Cloud9am = st.number_input("Cloud Cover at 9 AM (oktas)", min_value=0, max_value=8, value=3)
        Cloud3pm = st.number_input("Cloud Cover at 3 PM (oktas)", min_value=0, max_value=8, value=4)
        Temp9am = st.number_input("Temperature at 9 AM (°C)", min_value=-10.0, max_value=50.0, value=20.0)
        Temp3pm = st.number_input("Temperature at 3 PM (°C)", min_value=-10.0, max_value=50.0, value=28.0)
        RainToday = st.selectbox("Did it rain today?", ["No", "Yes"])
    
    RainToday = 1 if RainToday == "Yes" else 0
    WindGustDir = encode_wind_direction(WindGustDir)
    WindDir9am = encode_wind_direction(WindDir9am)
    WindDir3pm = encode_wind_direction(WindDir3pm)
    
    input_data = [MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed,
                  WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm,
                  Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday]
    
    if st.button("Predict"):
        try:
            prediction = rainfall_prediction(input_data)
            if prediction[0] == 0:
                prediction_text = "No rain will fall tommorrow"
                result_placeholder.success(prediction_text)
                st.success(prediction_text)
            else:
                prediction_text = f"Rain will fall tommorrow%"
                result_placeholder.error(prediction_text)
                st.success(prediction_text)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            result_placeholder.error("An error occurred during prediction. Please check input data.")

if __name__ == "__main__":
    main()
