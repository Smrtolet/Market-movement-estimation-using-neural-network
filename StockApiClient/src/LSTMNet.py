from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iexfinance.stocks import get_historical_data
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

# promjenjive varijable
start = datetime(2017, 1, 1)  # yyyy/mm/dd
end = datetime.today()
training_percent = 0.6  # [0, 1]
num_of_stocks = 1  # 1, 2 ili 3
hidden_layers_num = [2, 3] # array sa dvije vrijednosti: [2,3] provjerava samo mogućnosti sa dva skrivena sloja
hidden_layer_neurons = [10, 11]
epochs = 50
batch_size  = 32
dropout = 0.2
do_print = 0 # 1 za ispisivanje outputa

SPY_sc = MinMaxScaler(feature_range=(0, 1))
SPY_sv = MinMaxScaler(feature_range=(0, 1))
IWM_sc = MinMaxScaler(feature_range=(0, 1))
IWM_sv = MinMaxScaler(feature_range=(0, 1))
DIA_sc = MinMaxScaler(feature_range=(0, 1))
DIA_sv = MinMaxScaler(feature_range=(0, 1))

dataset_SPY = get_historical_data("SPY", start, end, output_format='pandas')
training_size = int(dataset_SPY.shape[0] * training_percent)

training_SPY_close = dataset_SPY.iloc[:training_size, 3:4].values
training_SPY_scaled_close = SPY_sc.fit_transform(training_SPY_close)

training_SPY_volume = dataset_SPY.iloc[:training_size, 4:5].values
training_SPY_scaled_volume = SPY_sv.fit_transform(training_SPY_volume)

if (num_of_stocks == 2):
    dataset_IWM = get_historical_data("IWM", start, end, output_format='pandas')

    training_IWM_close = dataset_IWM.iloc[:training_size, 3:4].values
    training_IWM_scaled_close = IWM_sc.fit_transform(training_IWM_close)

    training_IWM_volume = dataset_IWM.iloc[:training_size, 4:5].values
    training_IWM_scaled_volume = IWM_sv.fit_transform(training_IWM_volume)

if (num_of_stocks == 3):
    dataset_DIA = get_historical_data("DIA", start, end, output_format='pandas')

    training_DIA_close = dataset_DIA.iloc[:training_size, 3:4].values
    training_DIA_scaled_close = DIA_sc.fit_transform(training_DIA_close)

    training_DIA_volume = dataset_DIA.iloc[:training_size, 4:5].values
    training_DIA_scaled_volume = DIA_sv.fit_transform(training_DIA_volume)

# kreiraj matrice ulaznih i izlaznih vrijednosti
X_train = []
y_train = []
training_length = 60

for i in range(training_length, training_SPY_scaled_close.shape[0]):
    temp = []
    temp.extend(training_SPY_scaled_close[i - training_length:i, 0])
    temp.extend(training_SPY_scaled_volume[i - training_length:i, 0])
    if(num_of_stocks == 2):
        temp.extend(training_IWM_scaled_close[i-training_length:i, 0])
        temp.extend(training_IWM_scaled_volume[i-training_length:i, 0])

    if(num_of_stocks==3):
        temp.extend(training_DIA_scaled_close[i-training_length:i, 0])
        temp.extend(training_DIA_scaled_volume[i-training_length:i, 0])

    X_train.append(temp)
    y_train.append(training_SPY_scaled_close[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
# dodaje treću dimenziju matrici
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# input shape je redak koji pretstavlja jedan ulaz
for i in range(hidden_layer_neurons[0], hidden_layer_neurons[1]):
    for j in range(hidden_layers_num[0], hidden_layers_num[1]):
        regressor = Sequential()
        regressor.add(LSTM(units=i, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        for k in range(j-1):
            regressor.add(LSTM(units=i, return_sequences=True))
            # dropaj(postavi na nulu) 20% inputa tj. outputa iz LSTM sloja
            regressor.add(Dropout(dropout))
        regressor.add(LSTM(units=i))
        regressor.add(Dropout(dropout))
        regressor.add(Dense(units=1))
        regressor.compile(optimizer='adam', loss='mean_squared_error')
        print(X_train.shape, y_train.shape)
        regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose = do_print)
        model_json = regressor.to_json()
        with open("{}Layers_{}Neurons_model.json".format(j, i), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        regressor.save_weights("{}Layers_{}Neurons_model.h5".format(j, i))
        print(i, j);


# to load model
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
# test_SPY_close = dataset_SPY.iloc[:, 3:4].values
# test_SPY_volume = dataset_SPY.iloc[:, 4:5].values



# # test_IWM_close = dataset_IWM.iloc[training_size:, 3:4].values
# # test_IWM_volume = dataset_IWM.iloc[training_size:, 4:5].values
# #
# # test_DIA_close = dataset_DIA.iloc[training_size:, 3:4].values
# # test_DIA_volume = dataset_DIA.iloc[training_size:, 4:5].values
#
# test_SPY_scaled_close = SPY_sc.transform(test_SPY_close)
# test_SPY_scaled_volume = SPY_sv.transform(test_SPY_volume)
# # test_IWM_scaled_close = IWM_sc.transform(test_IWM_close)
# # test_IWM_scaled_volume = IWM_sv.transform(test_IWM_volume)
# # test_DIA_scaled_close = DIA_sc.transform(test_DIA_close)
# # test_DIA_scaled_volume = DIA_sv.transform(test_DIA_volume)
#
#
# X_test = []
# for i in range(training_length, test_SPY_scaled_close.shape[0]):
#     temp = []
#     temp.extend(test_SPY_scaled_close[i - training_length:i, 0])
#     temp.extend(test_SPY_scaled_volume[i - training_length:i, 0])
#     # temp.extend(test_IWM_scaled_close[i-training_length:i, 0])
#     # temp.extend(test_IWM_scaled_volume[i-training_length:i, 0])
#     # temp.extend(test_DIA_scaled_close[i-training_length:i, 0])
#     # temp.extend(test_DIA_scaled_volume[i-training_length:i, 0])
#     X_test.append(temp)
#
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# predicted_stock_price = regressor.predict(X_test)
# print(predicted_stock_price)
# predicted_stock_price = SPY_sc.inverse_transform(predicted_stock_price)
# Y_test = dataset_SPY.iloc[:, 3:4].values[training_length:test_SPY_scaled_close.shape[0], 0]
#
# plt.plot(Y_test, color='black', label='SPY Stock Price')
# plt.plot(predicted_stock_price, color='green', label='Predicted SPY Stock Price')
# plt.title('SPY Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('SPY Stock Price')
# plt.legend()
# plt.show()
