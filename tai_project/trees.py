#!/usr/bin/python3
import numpy as np
import pandas as pd 
from sklearn import tree
from sklearn.model_selection import train_test_split
from ast import literal_eval
import matplotlib.pyplot as plt
import pickle




# if we want to use the unprocessed data
df_input = pd.read_csv('data/input_data_2.csv')
df_input  = df_input.drop(columns=['datetime', 'number'])
df_input  = df_input.drop(columns=df_input.columns[0])
df_target = pd.read_csv('data/target_data_2.csv')
df_target = df_target.drop(columns=df_target.columns[0])

print(df_target.head())
print(df_input.head())
print(np.shape(df_input), np.shape(df_target))


# if we were to use the timestep data
# df_input  = pd.read_csv('data/input_data_1.csv')
# df_input  = df_input.drop(columns=df_input.columns[0])
# df_target = pd.read_csv('data/target_data_1.csv')
# df_target = df_target.drop(columns=df_target.columns[0])

arr_target = np.array(df_target)
arr_input  = np.array(df_input)

# splitting into 80, 10, 10
x_train, x_valtest, y_train, y_valtest = train_test_split(arr_input, arr_target, test_size = 0.2, random_state = 0, shuffle=True)
x_val, x_test, y_val, y_test = train_test_split(x_valtest, y_valtest, test_size= 0.5, random_state = 0, shuffle=True)


rmse = {"train_temp":[], "train_wind":[], "train_precip":[], "val_temp":[], "val_wind":[], "val_precip":[]}
tempMin = 100
windMin = 100
precipMin = 100
start, end = 1, 12
bestTree = [[], [], []]
for depth in range(start, end):
    temperatureTree = tree.DecisionTreeRegressor(max_depth=depth)
    windTree = tree.DecisionTreeRegressor(max_depth=depth)
    precipTree = tree.DecisionTreeRegressor(max_depth=depth)
    temperatureTree.fit(x_train, y_train[:, 0])
    windTree.fit(x_train, y_train[:, 2])
    precipTree.fit(x_train, y_train[:, 3])
    if np.sqrt(np.mean((temperatureTree.predict(x_val)-y_val[:, 0])**2)) < tempMin:
        bestTree[0] = temperatureTree
        tempMin = np.sqrt(np.mean((temperatureTree.predict(x_val)-y_val[:, 0])**2))
        print("Temp:", depth)
    if np.sqrt(np.mean((windTree.predict(x_val)-y_val[:, 2])**2)) < windMin:
        bestTree[1] = windTree
        windMin = np.sqrt(np.mean((windTree.predict(x_val)-y_val[:, 2])**2))
        print("Wind:", depth)
    if np.sqrt(np.mean((precipTree.predict(x_val)-y_val[:, 3])**2)) < precipMin:
        bestTree[2] = precipTree
        precipMin = np.sqrt(np.mean((precipTree.predict(x_val)-y_val[:, 3])**2))
        print("Precip:", depth)
    rmse["train_temp"].append((np.mean((temperatureTree.predict(x_train)-y_train[:, 0])**2)))
    rmse["val_temp"].append((np.mean((temperatureTree.predict(x_val)-y_val[:, 0])**2)))
    rmse["train_wind"].append((np.mean((windTree.predict(x_train)-y_train[:, 2])**2)))
    rmse["val_wind"].append((np.mean((windTree.predict(x_val)-y_val[:, 2])**2)))
    rmse["train_precip"].append((np.mean((precipTree.predict(x_train)-y_train[:, 3])**2)))
    rmse["val_precip"].append((np.mean((precipTree.predict(x_val)-y_val[:, 3])**2)))
    # best trees
    # temp: 11
    # wind: 13
    # precip: 7
pickle.dump(bestTree[0], open("TemperatureTree", "wb"))
pickle.dump(bestTree[1], open("WindTree", "wb"))
pickle.dump(bestTree[2], open("PrecipTree", "wb"))



plt.figure()
plt.plot(range(start, end), rmse["train_temp"], label='training')
plt.plot(range(start, end), rmse["val_temp"], label='validation')
plt.legend()
plt.title("Temperature")
plt.xlabel("Depth")
plt.ylabel("MSE")
plt.savefig('temptree.png')

plt.figure()
plt.plot(range(start, end), rmse["train_wind"], label='training')
plt.plot(range(start, end), rmse["val_wind"], label='validation')
plt.legend()
plt.title("Wind speed")
plt.ylabel("MSE")
plt.xlabel("Depth")
plt.savefig('windtree.png')

plt.figure()
plt.plot(range(start, end), rmse["train_precip"], label='training')
plt.plot(range(start, end), rmse["val_precip"], label='validation')
plt.legend()
plt.title("Precipitation")
plt.ylabel("MSE")
plt.xlabel("Depth")
plt.savefig('prectree.png')

# print("The final loss value is: ", np.mean((clf.predict(x_test) - y_test) ** 2))
print("For Temperature: ", np.mean((bestTree[0].predict(x_test) - y_test[:,0]) ** 2))
# print("For Winddirection: ", np.mean((clf.predict(x_test)[:,1] - y_test[:,1]) ** 2))
print("For Windspeed: ", np.mean((bestTree[1].predict(x_test) - y_test[:,2]) ** 2))
print("For Precipitation: ", np.mean((bestTree[2].predict(x_test) - y_test[:,3]) ** 2))

# load in forecastdata 

df_forecast = pd.read_csv('forecast/training_input_2.csv')
print(df_forecast.head())
df_forecast  = df_forecast.drop(columns = df_forecast.columns[0])
arr_forecast = np.array(df_forecast)
t_pred = bestTree[0].predict(arr_forecast)
w_pred = bestTree[1].predict(arr_forecast)
p_pred = bestTree[2].predict(arr_forecast)
print(np.shape(t_pred), np.shape(t_pred), np.shape(t_pred))


total_pred = np.zeros((1000,3))
total_pred[:,0] = t_pred
total_pred[:,1] = w_pred
total_pred[:,2] = p_pred

print(np.shape(total_pred), " is the shape")
df_pred = pd.DataFrame(total_pred, columns=[['t2m', 'wind', 'tp6']])

df_pred.to_csv('forecast/prediction.csv')

# pd.DataFrame(total_pred).to_csv('forecast/prediction.csv')

# FOR THE FIRST DATASET: 
# convert back into usual format; first 20 values, next 20 values and so on and save prediction 
# new_arr = np.zeros((1000,3))
# for i in range(len(total_pred)):
#     new_arr[i, :] = total_pred[int((i%20)*20 + np.floor(i/20))]
# pd.DataFrame(new_arr).to_csv('forecast/prediction.csv')

# print(np.shape(total_pred), np.shape(t_pred))



