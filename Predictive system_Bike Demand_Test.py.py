import sklearn
import numpy as np
import pickle

file_path = r'C:\Users\HP\OneDrive\Desktop\Deploying Machine learning Model\rfr_model.csv'
loaded_model = pickle.load(open(file_path, 'rb'))

input_data = {"Temperature(°C)": -6.0,
    "Humidity(%)": 36,
    "Wind_speed_(m/s)": 2.3,
    "Visibility_(10m)": 2000,
    "Dew_point_temperature(°C)": -18.6,
    "Solar_Radiation_(MJ/m2)": 0.00,
    "Rainfall(mm)": 0.0,
    "Snowfall_(cm)": 0.0,
    "Seasons": 3,
    "Holiday": 1,
    "Functioning_Day": 1,
    "Cluster": 2  # Assuming cluster is pre-determined
} 
input_data_as_numpy = np.array(list(input_data.values()))
input_data_reshaped = input_data_as_numpy.reshape(1, -1)
# Make prediction
prediction = loaded_model.predict(input_data_reshaped)

# # Print result
print(f"Predicted Rented Bike Count: {int(prediction[0])}")


