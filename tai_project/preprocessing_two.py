from pp_functions import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''


'''

# load in prepared data for preprocessing 
df_true = pd.read_csv('data/target_data_1.csv')
df_input = pd.read_csv('data/forecastdata.csv')


# turn them into np.arrays, drop columns that are not interesting for training 
input_arr = np.array(df_input.drop(columns=['datetime', 'number']))
target_arr = np.array(df_true.drop(columns=['datetime', 'local_datetime']))


# this function creates the input dataset, it takes numpy arrays in, gives numpy arrays out
training_input, training_target = makeTrainingSetOne(input_arr, target_arr)

# create pandas dataframes out of np arrays 
df_training_input = pd.DataFrame(training_input, columns=["T(t-1)", "WD(t-1)", "WS(t-1)", "P(t-1)", "T(t)", "WD(t)", "WS(t)", "P(t)", "T(t+1)", "WD(t+1)", "WS(t+1)", "P(t+1)"])
df_training_target = pd.DataFrame(training_target, columns=["T(t)", "WD(t)", "WS(t)", "P(t)"])


# in order to get rid of the NaN values, fuse them together, use dropna() and separate them again. This way, we get 2 arrays of the same shape back. 
# We have to do this because there are no NaN values in target, but some in input, and the respective lines in target would not be dropped

df_training_input[['temp_true', 'wind_direction_true', 'wind_speed_true', 'precip_quantity_6hour_true']] = df_training_target[['T(t)', 'WD(t)', 'WS(t)', 'P(t)']]
# drop the NaN values
df_training_input = df_training_input.dropna() 
# create new dataframe
df_training_target = pd.DataFrame()
df_training_target[['T(t)', 'WD(t)', 'WS(t)', 'P(t)']] = df_training_input[['temp_true', 'wind_direction_true', 'wind_speed_true', 'precip_quantity_6hour_true']]
df_training_input = df_training_input.drop(columns=['temp_true', 'wind_direction_true', 'wind_speed_true', 'precip_quantity_6hour_true'])

# finally, save preprocessed data 
df_training_input.to_csv("data/input_data_1.csv")
df_training_target.to_csv("data/target_data_1.csv")

print("check")






