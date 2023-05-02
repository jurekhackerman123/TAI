import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
import random
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
# k nearest neighbours
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

'''
We are trying out the machine learning with the k nearest neighbours method 
'''

df_target = pd.read_csv('data/target_data_1.csv')
df_target = df_target.drop(columns=df_target.columns[0])

df_input  = pd.read_csv('data/input_data_1.csv')
df_input  = df_input.drop(columns=df_input.columns[0])



# # try leaving out wind_direction as a prediction
# df_target = df_target.drop(columns=['WD(t)'])

arr_target = np.array(df_target)
arr_input  = np.array(df_input)

# scale input and target so that they have values between 0 and 1 
# scaler_features = MinMaxScaler((0,1))
# arr_input = scaler_features.fit_transform(arr_input)

# scaler_target = MinMaxScaler((0,1))
# arr_target = scaler_target.fit_transform(arr_target)

print(np.shape(arr_input))
print(np.shape(arr_target))


print(df_target.head())
# before using any machine learning on the data, we set mean = 1 and std = 0 for the input features, using standardscaler
scaler = StandardScaler()
input = scaler.fit_transform(arr_input)

# splitting into 80, 10, 10
x_train, x_valtest, y_train, y_valtest = train_test_split(arr_input, arr_target, test_size = 0.2, random_state = 0, shuffle=True)
x_val, x_test, y_val, y_test = train_test_split(x_valtest, y_valtest, test_size= 0.5, random_state = 0, shuffle=True)


# number of neighbours 
no_nb = 10

# Neigh = KNeighborsRegressor(n_neighbors=no_nb)
# Neigh.fit(x_train, y_train)
# Neigh.predict(x_test)



loss_train = []
loss_val   = []

loss_train_temp = []
loss_train_wind = []
loss_train_prec = []
loss_train_windd = []

loss_val_temp   = []
loss_val_wind   = []
loss_val_prec   = []
loss_val_windd   = []

max_number_epochs = 15
hls = 20

for i in range(1, no_nb):
    # train model for different number of neighbours 
    temp_model = KNeighborsRegressor(n_neighbors=i)
    wind_model = KNeighborsRegressor(n_neighbors=i)
    precip_model = KNeighborsRegressor(n_neighbors=i)
    temp_model.fit(x_train, y_train[:,0])
    wind_model.fit(x_train, y_train[:,2])
    precip_model.fit(x_train, y_train[:,3])
    y_pred_train_temp = temp_model.predict(x_train)
    y_pred_val_temp   = temp_model.predict(x_val)
    y_pred_train_wind = wind_model.predict(x_train)
    y_pred_val_wind   = wind_model.predict(x_val)
    y_pred_train_precip = precip_model.predict(x_train)
    y_pred_val_precip   = precip_model.predict(x_val)



    mse_train_temp = np.mean((y_pred_train_temp - y_train[:,0]) ** 2)
    mse_train_wind = np.mean((y_pred_train_wind - y_train[:,2]) ** 2)
    mse_train_prec = np.mean((y_pred_train_precip - y_train[:,3]) ** 2)
    # mse_train_windd = np.mean((y_pred_train[:,1] - y_train[:,1]) ** 2)

    mse_val_temp   = np.mean((y_pred_val_temp - y_val[:,0])**2)
    mse_val_wind   = np.mean((y_pred_val_wind - y_val[:,2])**2)
    mse_val_prec   = np.mean((y_pred_val_precip - y_val[:,3])**2)
    # mse_val_windd   = np.mean((y_pred_val[:,1] - y_val[:,1])**2)

    loss_train_temp.append(mse_train_temp)
    loss_train_wind.append(mse_train_wind)
    loss_train_prec.append(mse_train_prec)
    # loss_train_windd.append(mse_train_windd)

    loss_val_temp.append(mse_val_temp)
    loss_val_wind.append(mse_val_wind)
    loss_val_prec.append(mse_val_prec)
    # loss_val_windd.append(mse_val_windd)




plt.figure()
plt.plot(loss_train_temp, label="train-mse")
plt.plot(loss_val_temp, label="val-mse")
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Temperature MSE")
plt.savefig('pics/temperature_mse_knn.png')

plt.figure()
plt.plot(loss_train_wind, label="train-mse")
plt.plot(loss_val_wind, label="val-mse")
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Wind MSE")
plt.savefig('pics/windspeed_mse_knn.png')

plt.figure()
plt.plot(loss_train_prec, label="train-mse")
plt.plot(loss_val_prec, label="val-mse")
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Precipitation MSE")
plt.savefig('pics/precip_mse_knn.png')


print("For Temperature: ", np.mean((temp_model.predict(x_test) - y_test[:,0]) ** 2))
print("For Windspeed: ", np.mean((wind_model.predict(x_test) - y_test[:,2]) ** 2))
print("For Precipitation: ", np.mean((precip_model.predict(x_test) - y_test[:,3]) ** 2))



# forecast 
exit()


# load in forecast training input 
df_forecast = pd.read_csv('forecast/training_input.csv')
print(df_forecast.head())
df_forecast  = df_forecast.drop(columns = df_forecast.columns[0])
arr_forecast = np.array(df_forecast)
t_pred = Neigh.predict(arr_forecast)[:,0]
w_pred = Neigh.predict(arr_forecast)[:,1]
p_pred = Neigh.predict(arr_forecast)[:,2]
total_pred = Neigh.predict(arr_forecast)

pd.DataFrame(total_pred).to_csv('forecast/prediction.csv')

# convert back into usual format 
# first 20 values, next 20 values and so on and save prediction 
new_arr = np.zeros((1000,3))
for i in range(len(total_pred)):
    new_arr[i, :] = total_pred[int((i%20)*20 + np.floor(i/20))]

pd.DataFrame(new_arr).to_csv('forecast/prediction.csv')

print(np.shape(total_pred), np.shape(t_pred))




