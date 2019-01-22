import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from iexfinance.stocks import get_historical_data

start = datetime(2015, 1, 1)
end = datetime(2019,1,1)

StockShortname ="NFLX"
dataset_A = get_historical_data(StockShortname, start, end, output_format='pandas')
trainset = dataset_A.iloc[:,3:].values

incrementSet = trainset.copy()[:,0]
for i in range(0,60):
    incrementSet[i]= 0
for i in range(60,1006):
    incrementSet[i] = trainset[i][0]-trainset[i-1][0]


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
bc = MinMaxScaler(feature_range = (0, 1))
ic = MinMaxScaler(feature_range = (0, 1))
training_set_scaled =sc.fit_transform(trainset)
tmp = bc.fit_transform(trainset[:,0].reshape(-1,1))
increment_scaled = ic.fit_transform(incrementSet.reshape(-1,1))

X_train = []
y_train = []
X_test = []
y_test = []
for i in range(60, 806):
    X_train.append(training_set_scaled[i-60:i])
    y_train.append(increment_scaled[i][0])
for i in range(806, 1006):
    X_test.append(training_set_scaled[i-60:i])
    y_test.append(increment_scaled[i][0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test =  np.array(X_test), np.array(y_test)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 2))



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout



# Init RNN
regressor = Sequential()

# 1st LSTM layer
regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 2)))
regressor.add(Dropout(0.2))

# 2nd LSTM layer
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2))

# 3rd LSTM layer
regressor.add(LSTM(units = 60, return_sequences = True))
regressor.add(Dropout(0.2))

# 4th LSTM layer
regressor.add(LSTM(units = 60))
regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units = 1))


#training
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 20, batch_size = 30)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = ic.inverse_transform(predicted_stock_price)
for i in range(0,len(predicted_stock_price)):
    predicted_stock_price[i] = predicted_stock_price[i] + trainset[806+i][0]
    
real_stock_price = trainset.copy()[806:,0:1]


# Visualising the results
plt.figure(figsize=(20,15))
plt.plot(real_stock_price, color = 'red', label = 'Real '+ StockShortname +' Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted '+StockShortname+' Stock Price')
plt.title(''+StockShortname+' Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel(''+StockShortname+' Stock Price')
plt.legend()
plt.show()


print("MSE is: ",sum((real_stock_price-predicted_stock_price)*(real_stock_price-predicted_stock_price))/len(real_stock_price))