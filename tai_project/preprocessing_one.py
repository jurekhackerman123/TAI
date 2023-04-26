from pp_functions import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''
We now created two datasets, one for the targetdata, one for the inputdata. 
What we want to do now is create an input dataset for the training. The first attempt looks as follows: 
We are going to, for each set of 20 values (5 days forecast for each day) import the value for the temperature, precipitation, windspeed and -angle for the timesteps t, t-1 and t+1 
where t-1 and t+1 correspond to the values at times - or + 6h. Where the first of the 20 is t,t,t+1 and the last one t-1,t
We are going to create a 12 x (len(input)) - dimensional dataset with the rows consisting of T(t-1) P(t-1) WD(t-1) WS(t-1) T(t) ... 
and the respective target dataset looks as follows: 
T(t) P(t) WD(t) WS(t)
T(t+1) P(t+1) WD(t+1) WS(t+1)
T(t+2) P(t+2) WD(t+2) WS(t+2)
.
.
.
T(t+19) P(t+19) WD(t+19) WS(t+19)
T(t+1) P(t+1) WD(t+1) WS(t+1)           // here the five day forecast for the second day starts 
and so on

'''

df_true = pd.read_csv('data/true_data.csv')
df_input = pd.read_csv('data/forecastdata.csv')



input_arr = np.array(df_input.drop(columns=['datetime', 'number']))
target_arr = np.array(df_true.drop(columns=['datetime', 'local_datetime']))



training_input, training_target = makeTrainingSetOne(input_arr, target_arr)

df_training_input = pd.DataFrame(training_input, columns=["T(t-1)", "WD(t-1)", "WS(t-1)", "P(t-1)", "T(t)", "WD(t)", "WS(t)", "P(t)", "T(t+1)", "WD(t+1)", "WS(t+1)", "P(t+1)"])
df_training_target = pd.DataFrame(training_target, columns=["T(t)", "WD(t)", "WS(t)", "P(t)"])
df_training_input.to_csv("data/input_data_1.csv")
df_training_target.to_csv("data/target_data_1.csv")

print("check")