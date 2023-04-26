
import numpy as np
import pandas as pd
import math 
# from sklearn import tree
# from sklearn.model_selection import train_test_split
# import pickle
import matplotlib.pyplot as plt


def wind_direction(long, lat):
    vectorAngle = np.vectorize(Angle)
    wind_direction = vectorAngle(long, lat)
    return wind_direction

def Angle(long, lat):
    if long > 0 and lat > 0:
        long = abs(long)
        lat = abs(lat)
        return np.arctan(long / lat)* 180/np.pi + 180

    if long < 0 and lat < 0:
        long = abs(long)
        lat = abs(lat)
        return np.arctan(long / lat)* 180/np.pi
    
    if long < 0 and lat > 0:
        long = abs(long)
        lat = abs(lat)
        return 180 - np.arctan(long / lat)* 180/np.pi
    
    if long > 0 and lat < 0:
        long = abs(long)
        lat = abs(lat)
        return 360 - np.arctan(long / lat)* 180/np.pi


def calcRh(d, t):
    return np.array(100*(np.exp((d * 17.625) / (234.04 + d)) / 
                           (np.exp((t * 17.625) / (234.04  + t)))))

def formatDate(dataframe):
    dataframe["forecast_date"] = str(dataframe["time"]).split(" ")[0]
    dataframe["horizon"] = str(dataframe["time"]).split(" ")[1].split(":")[0]  + "hour"

def interpolate(frames):
    'linearly interpolate values for every full hour'
    newFrames = []
    for ele in frames:
        insert = 6
        ele.index = range(0, (len(ele)) * insert, insert)
        ele = ele.reindex(index=range((len(ele)-1)*insert + 1))
        ele = ele.interpolate()
        newFrames.append(ele)
    surInterpolated = pd.concat(newFrames, ignore_index=True)
    print(surInterpolated.shape)
    return surInterpolated


# in order to have the target precipitation data also in timesteps of 6hrs, sum up the first 6 values of precip (each corresponding to the total precip )
def accumulate_precip(precip_data):
    accumulated = pd.Series(precip_data.rolling(6).sum())
    return accumulated


def makeTrainingSetOne(Input_arr, Target_arr):
    training_input = np.zeros((len(Input_arr), 12))
    training_target = np.zeros((len(Input_arr), 4))
    for i in range(len(Input_arr)):
        if i%10000 == 0:
            print("step")
        if i%20 == 0:
            training_input[i, 0:4] = Input_arr[i, :]
            training_input[i, 4:8] = Input_arr[i, :]
            training_input[i, 8:12] = Input_arr[i+1, :]
        elif i%20 == 19: 
            training_input[i, 0:4] = Input_arr[i-1, :]
            training_input[i, 4:8] = Input_arr[i, :]
            training_input[i, 8:12] = Input_arr[i, :]
        else:
            training_input[i, 0:4] = Input_arr[i-1, :]
            training_input[i, 4:8] = Input_arr[i, :]
            training_input[i, 8:12] = Input_arr[i+1, :]
        # after each five days, we reset to the next day
        # after one ensemble is over, we start over 
        training_target[i, :] = Target_arr[int(i%20+np.floor(i/20))%(244), :]
    return training_input, training_target



def quantiles(predictions):
    result = pd.DataFrame(columns=["forecast_date", "target", "horizon", "q0.025", "q0.25", "q0.5", "q0.75", "q0.975"])
    quants = [0.025, 0.25, 0.5, 0.75, 0.975]
    for i in range(0, 20):
        horizon = i * 6 + 6
        quantsTimestepTemp = []
        quantsTimestepWind = []
        quantsTimestepPrecip = []
        for quant in quants:
            quantsTimestepTemp.append(np.quantile(predictions["t2m"][i::20], quant))
            quantsTimestepWind.append(np.quantile(predictions["wind"][i::20], quant))
            quantsTimestepPrecip.append(np.quantile(predictions["precip"][i::20], quant))
        result = result.append(pd.Series(["2023-04-15", "t2m", str(horizon) + " hour", *quantsTimestepTemp], 
                                         index=["forecast_date", "target", "horizon", "q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]),
                                         ignore_index=True)
        result = result.append(pd.Series(["2023-04-15", "wind", str(horizon) + " hour", *quantsTimestepWind], 
                               index=["forecast_date", "target", "horizon", "q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]), 
                               ignore_index=True)
        result = result.append(pd.Series(["2023-04-15", "precip", str(horizon) + " hour", *quantsTimestepPrecip], 
                               index=["forecast_date", "target", "horizon", "q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]), 
                               ignore_index=True)
    return result









