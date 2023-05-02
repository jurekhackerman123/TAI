import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pp_functions import *
# import pp_functions

# in this file, we basically prepare the data for the preprocessing 
# we filter out the columns of interest, and bring everything in the right format 

# load in data 
df_precip = pd.read_csv('ECMWF_2017_2018_precip.csv')
df_surface = pd.read_csv('ECMWF_2017_2018_surface.csv')

df_2017_true = pd.read_csv('synop_2017_March_June.csv')
df_2018_true = pd.read_csv('synop_2018_March_June.csv')

# only keep the columns we need 
df_precip = df_precip[['number', 'valid_time', 'tp6']]
df_surface = df_surface[['number', 'valid_time', 'u10', 'v10', 't2m']]


# calculate the wind speed and direction
df_surface['wind_speed'] = np.sqrt(df_surface['u10']**2 + df_surface['v10']**2)
df_surface['wind_direction'] = wind_direction(df_surface['u10'], df_surface['v10'])


# in precip, the first 00:00:00 is missing for every day, so we delete it for df_surface 
df_surface = df_surface[df_surface.index % 21 != 0]
df_surface = df_surface.reset_index(drop=True)
# df_precip and df_surface are now equally long

# rename valid_time to datetime to make everything consistent
df_precip = df_precip.rename(columns={'valid_time':'datetime'})
df_surface = df_surface.rename(columns={'valid_time':'datetime'})

# create one dataframe containing the input
df_input = df_surface[['datetime', 'number', 't2m', 'wind_direction', 'wind_speed']]

# calculate precip in mm and temperatur in C and save input dataset in forecastdata.csv
df_input['tp6'] =  df_precip[['tp6']]*1000
df_input['t2m'] = df_input['t2m']-273.15

# and save input data as .csv file 
df_input.to_csv('data/forecastdata.csv', index = False)


# now for the targetdata
# keep columns we need 
df_2017_true = df_2017_true[['datetime', 'temp', 'wind_direction', 'wind_speed', 'precip_quantity_1hour', 'local_datetime']]
df_2018_true = df_2018_true[['datetime', 'temp', 'wind_direction', 'wind_speed', 'precip_quantity_1hour', 'local_datetime']]

# fuse the two years together 
df_true = pd.concat([df_2017_true, df_2018_true], ignore_index=True, axis=0)

# keep only every second value, every value is in there twice. 
df_true = df_true.iloc[::2, :]

# add the values of the precip up
# what we do here is we shift the precipdata one index up and sum the first six values together. We only keep the values for 00:00:00, 06:00:00 in steps of 6h
# and save the dataset 
df_true['precip_quantity_6hour'] = np.append(np.array([0]), accumulate_precip(df_true['precip_quantity_1hour']))[:-1]
df_true = df_true.iloc[::6, :]

# we don't need this column anymore 
df_true = df_true.drop(columns=['precip_quantity_1hour'])

# drop first row
df_true = df_true.tail(-1)

# save target data 
df_true.to_csv('data/true_data.csv', index = False)

