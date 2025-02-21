import numpy as np
import pickle
import streamlit as st
import pandas as pd 

file_path = r'C:\Users\HP\OneDrive\Desktop\Deploying Machine learning Model\rfr_model.csv'
loaded_model = pickle.load(open(file_path, 'rb'))


# Creating a function for the prediction
def Bike_Demand_Prediction(input_data):
    input_data_as_numpy = input_data.values
    input_data_reshaped = input_data_as_numpy.reshape(1, -1)
    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_data_reshaped)
    return f"The forecast for Bike Demand is : {int(prediction[0])}"

def main():
    # Title of webpage
    st.title("Bike Demand Prediction Web App")

    st.markdown("Enter available information below:")
    # Getting the input data from users
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=40.0, value=20.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=10.0, value=2.0)
    visibility = st.number_input("Visibility (10m)", min_value=0, max_value=2000, value=1000)
    dew_point = st.number_input("Dew Point Temperature (°C)", min_value=-20.0, max_value=30.0, value=10.0)
    solar_radiation = st.number_input("Solar Radiation (MJ/m²)", min_value=0.0, max_value=5.0, value=1.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=50.0, value=0.0)
    snowfall = st.number_input("Snowfall (cm)", min_value=0.0, max_value=10.0, value=0.0)

    seasons = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
    holiday = st.selectbox("Holiday", ["No Holiday", "Holiday"])
    functioning_day = st.selectbox("Functioning Day", ["Yes", "No"])
    cluster = st.selectbox("Cluster", ["kmeans1", "kmeans2", "kmeans3"])

    # Encode categorical variables
    seasons_dict = {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3}
    holiday_dict = {"No Holiday": 0, "Holiday": 1}
    functioning_day_dict = {"Yes": 1, "No": 0}
    cluster_dict = {"kmeans1": 1, "kmeans2": 2, "kmeans3": 3}

    seasons = seasons_dict[seasons]
    holiday = holiday_dict[holiday]
    functioning_day = functioning_day_dict[functioning_day]
    cluster = cluster_dict[cluster]

    # Convert inputs into DataFrame
    input_data = pd.DataFrame(
        [[temperature, humidity, wind_speed, visibility, dew_point, solar_radiation,
          rainfall, snowfall, seasons, holiday, functioning_day, cluster]],
        columns=[
            "Temperature(°C)", "Humidity(%)", "Wind_speed_(m/s)", "Visibility_(10m)",
            "Dew_point_temperature(°C)", "Solar_Radiation_(MJ/m2)", "Rainfall(mm)",
            "Snowfall_(cm)", "Seasons", "Holiday", "Functioning_Day", "Cluster"
        ]
    )

    # Predict button
    if st.button("Bike Demand Forecast"):
        forecast = Bike_Demand_Prediction(input_data)
        st.success(forecast)

if __name__ == '__main__':
    main()
