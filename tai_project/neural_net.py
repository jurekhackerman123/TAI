import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
import random
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
# linear regression
from sklearn.linear_model import LinearRegression
# neural net
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


df_target = pd.read_csv('data/target_data_1.csv')
df_target = df_target.drop(columns=df_target.columns[0])

df_input  = pd.read_csv('data/input_data_1.csv')
df_input  = df_input.drop(columns=df_input.columns[0])

# in order to get rid of the NaN values, fuse them together, use dropna() and separate them again. This way, we get 2 arrays of the same shape back. 
# We have to do this because there are no NaN values in target, but some in input, and the respective lines in target would not be dropped

df_input[['temp_true', 'wind_direction_true', 'wind_speed_true', 'precip_quantity_6hour_true']] = df_target[['T(t)', 'WD(t)', 'WS(t)', 'P(t)']]
# drop the NaN values
df_input = df_input.dropna() 
# create new dataframe
df_target = pd.DataFrame()
df_target[['T(t)', 'WD(t)', 'WS(t)', 'P(t)']] = df_input[['temp_true', 'wind_direction_true', 'wind_speed_true', 'precip_quantity_6hour_true']]
df_input = df_input.drop(columns=['temp_true', 'wind_direction_true', 'wind_speed_true', 'precip_quantity_6hour_true'])

# try leaving out wind_direction as a prediction
df_target = df_target.drop(columns=['WD(t)'])

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


print(np.shape(x_train), np.shape(x_val), np.shape(x_test), np.shape(y_train), np.shape(y_val), np.shape(y_test))


'''
Linear regression
'''

# lin_reg = LinearRegression()
# lin_reg.fit(x_train, y_train)

# w = lin_reg.coef_
# b = lin_reg.intercept_

# print("w = ", w)
# print("b=", b)



'''
Neural net
'''


clf = MLPRegressor(solver='adam', alpha= 1e-5, hidden_layer_sizes=(7,6), random_state=0)

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

for i in range(1, max_number_epochs):
    # clf = MLPRegressor(solver='adam', alpha= 1e-5, hidden_layer_sizes=(10,10,10), random_state=0, max_iter = i)
    # partial fit only works for alan, bc apparently, it's stochastic
    clf.partial_fit(x_train, y_train)
    # clf.fit(x_train, y_train)
    y_pred_train = clf.predict(x_train)
    y_pred_val   = clf.predict(x_val)
    # y_pred_test   = clf.predict(x_test)     

    # 1 is wind_direction

    mse_train_temp = np.mean((y_pred_train[:,0] - y_train[:,0]) ** 2)
    mse_train_wind = np.mean((y_pred_train[:,1] - y_train[:,1]) ** 2)
    mse_train_prec = np.mean((y_pred_train[:,2] - y_train[:,2]) ** 2)
    # mse_train_windd = np.mean((y_pred_train[:,1] - y_train[:,1]) ** 2)

    mse_val_temp   = np.mean((y_pred_val[:,0] - y_val[:,0])**2)
    mse_val_wind   = np.mean((y_pred_val[:,1] - y_val[:,1])**2)
    mse_val_prec   = np.mean((y_pred_val[:,2] - y_val[:,2])**2)
    # mse_val_windd   = np.mean((y_pred_val[:,1] - y_val[:,1])**2)
    
    mse_train   = np.mean((y_pred_train - y_train)**2)
    mse_val     = np.mean((y_pred_val - y_val)**2)

    loss_train_temp.append(mse_train_temp)
    loss_train_wind.append(mse_train_wind)
    loss_train_prec.append(mse_train_prec)
    # loss_train_windd.append(mse_train_windd)

    loss_val_temp.append(mse_val_temp)
    loss_val_wind.append(mse_val_wind)
    loss_val_prec.append(mse_val_prec)
    # loss_val_windd.append(mse_val_windd)
    
    loss_train.append(mse_train)
    loss_val.append(mse_val)

'''
max_dim = 10
number_epochs = 15

for i in range(1, max_dim):
    # clf = MLPRegressor(solver='adam', alpha= 1e-5, hidden_layer_sizes=(10,10,10), random_state=0, max_iter = i)
    clf = MLPRegressor(solver='adam', alpha= 1e-5, hidden_layer_sizes=(i,i), random_state=0, max_iter = number_epochs)

    clf.fit(x_train, y_train)
    y_pred_train = clf.predict(x_train)
    y_pred_val   = clf.predict(x_val)
    # y_pred_test   = clf.predict(x_test)     

    # 1 is wind_direction

    mse_train_temp = np.mean((y_pred_train[:,0] - y_train[:,0]) ** 2)
    mse_train_wind = np.mean((y_pred_train[:,2] - y_train[:,2]) ** 2)
    mse_train_prec = np.mean((y_pred_train[:,3] - y_train[:,3]) ** 2)
    mse_train_windd = np.mean((y_pred_train[:,1] - y_train[:,1]) ** 2)

    mse_val_temp   = np.mean((y_pred_val[:,0] - y_val[:,0])**2)
    mse_val_wind   = np.mean((y_pred_val[:,2] - y_val[:,2])**2)
    mse_val_prec   = np.mean((y_pred_val[:,3] - y_val[:,3])**2)
    mse_val_windd   = np.mean((y_pred_val[:,1] - y_val[:,1])**2)
    
    mse_train   = np.mean((y_pred_train - y_train)**2)
    mse_val     = np.mean((y_pred_val - y_val)**2)

    loss_train_temp.append(mse_train_temp)
    loss_train_wind.append(mse_train_wind)
    loss_train_prec.append(mse_train_prec)
    loss_train_windd.append(mse_train_windd)

    loss_val_temp.append(mse_val_temp)
    loss_val_wind.append(mse_val_wind)
    loss_val_prec.append(mse_val_prec)
    loss_val_windd.append(mse_val_windd)
    
    loss_train.append(mse_train)
    loss_val.append(mse_val)

'''



plt.figure()
plt.plot(loss_train_temp, label="train-mse")
plt.plot(loss_val_temp, label="val-mse")
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Temperature MSE")
plt.savefig('pics/temperature_mse.png')

plt.figure()
plt.plot(loss_train_wind, label="train-mse")
plt.plot(loss_val_wind, label="val-mse")
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Wind MSE")
plt.savefig('pics/windspeed_mse.png')

# plt.figure()
# plt.plot(loss_train_windd, label="train-mse")
# plt.plot(loss_val_windd, label="val-mse")
# plt.legend()
# plt.xlabel("Number of epochs")
# plt.ylabel("Wind direction MSE")
# plt.savefig('pics/winddirection_mse.png')

plt.figure()
plt.plot(loss_train_prec, label="train-mse")
plt.plot(loss_val_prec, label="val-mse")
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Precipitation MSE")
plt.savefig('pics/precip_mse.png')

plt.figure()
plt.plot(loss_train, label="train-mse")
plt.plot(loss_val, label="val-mse")
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Total MSE")
plt.savefig('pics/total_mse.png')


print("The final loss value is: ", np.mean((clf.predict(x_test) - y_test) ** 2))
print("For Temperature: ", np.mean((clf.predict(x_test)[:,0] - y_test[:,0]) ** 2))
# print("For Winddirection: ", np.mean((clf.predict(x_test)[:,1] - y_test[:,1]) ** 2))
print("For Windspeed: ", np.mean((clf.predict(x_test)[:,1] - y_test[:,1]) ** 2))
print("For Precipitation: ", np.mean((clf.predict(x_test)[:,2] - y_test[:,2]) ** 2))

# load in forecast training input 

df_forecast = pd.read_csv('forecast/training_input.csv')
print(df_forecast.head())
df_forecast  = df_forecast.drop(columns = df_forecast.columns[0])
arr_forecast = np.array(df_forecast)
t_pred = clf.predict(arr_forecast)[:,0]
w_pred = clf.predict(arr_forecast)[:,1]
p_pred = clf.predict(arr_forecast)[:,2]
total_pred = clf.predict(arr_forecast)

pd.DataFrame(total_pred).to_csv('forecast/prediction.csv')

new_arr = np.zeros((1000,3))

# convert back into usual format 
for i in range(len(total_pred)):
    new_arr[i, :] = total_pred[int((i%20)*20 + np.floor(i/20))]

pd.DataFrame(new_arr).to_csv('forecast/prediction.csv')

print(np.shape(total_pred), np.shape(t_pred))




