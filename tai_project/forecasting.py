import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
# from neural_net import * 
from pp_functions import *

# this file puts the forecastdata in the right format in order to make predictions with the machine learning models 

df_forecast = pd.read_csv('forecast/Forecast_29_04_2023.csv')

df_forecast['wind_direction'] = wind_direction(df_forecast['u10'], df_forecast['v10'])
df_forecast['wind_speed'] = np.sqrt(df_forecast['u10']**2 + df_forecast['v10']**2)
df_forecast['t2m'] = df_forecast['t2m']-273.15

df_forecast = df_forecast[['t2m', 'wind_direction', 'wind_speed', 'tp']]

# drop first 50 rows, because we have these for some reason... 
df_forecast = df_forecast.tail(df_forecast.shape[0] - 50)

print(np.shape(df_forecast))
arr_forecast = np.array(df_forecast)
new_arr = np.zeros((1000,4))


for i in range(len(df_forecast)):
    print(i)
    new_arr[i, 0:4] = arr_forecast[int((i%20)*50+np.floor(i/20)), :]




input_arr, no_array = makeTrainingSetOne(new_arr, new_arr)

df_new = pd.DataFrame(input_arr, columns=["T(t-1)", "WD(t-1)", "WS(t-1)", "P(t-1)", "T(t)", "WD(t)", "WS(t)", "P(t)", "T(t+1)", "WD(t+1)", "WS(t+1)", "P(t+1)"])

df_new.to_csv('forecast/training_input.csv')
df_forecast.to_csv('forecast/training_input_2.csv')


