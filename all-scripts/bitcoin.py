# Recurrent Neural Networks and Bitcoins

### Marco Tavora

## Introduction

'''
Nowadays most articles about bitcoin are speculative. Analyses based on strong technical foundations are rare.

My goal in this project is to build predictive models for the price of Bitcoins and other cryptocurrencies. To accomplish that, I will:
- Use first Long Short-Term Memory recurrent neural networks (LSTMs) for predictions;
- I will then study correlations between altcoins;
- The third step will be to repeat the first analysis using traditional time series.

For a thorough introduction to Bitcoins you can check this [book](https://github.com/bitcoinbook/bitcoinbook). For LSTM [this](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) is a great source.
'''

## Importing libraries

import numpy as np
import pandas as pd
import statsmodels.api as sm
import aux_functions as af 
from scipy import stats
import keras
import pickle
import quandl
from keras.models import Sequential
from keras.layers import Activation, Dense,LSTM,Dropout
from sklearn.metrics import mean_squared_error
from math import sqrt
from random import randint
from keras import initializers
import datetime
from datetime import datetime
from matplotlib import pyplot as plt
import json
import requests
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" # see the value of multiple statements at once.
pd.set_option('display.max_columns', None)

## Data
### There are several ways to fetch historical data for Bitcoins (or any other cryptocurrency). Here are a few examples:

###  Flat files
'''
If a flat file already available, we can just read it using `pandas`. For example, one of the datasets that will be used in this notebook can be found [here](https://www.kaggle.com/mczielinski/bitcoin-historical-data/data). This data is from *bitFlyer* a Bitcoin exchange and marketplace.
'''
df = pd.read_csv('bitcoin_data.csv')
'''
This dataset from [Kaggle](https://www.kaggle.com/mczielinski/bitcoin-historical-data/kernels) contains, for the time period of 01/2012-03/2018, minute-to-minute updates of the open, high, low and close prices (OHLC), the volume in BTC and indicated currency, and the [weighted bitcoin price](https://github.com/Bitcoin-Foundation-Italia/bitcoin-wp). 
'''
lst_col = df.shape[1]-1
df.iloc[:,lst_col].plot(lw=2, figsize=(15,5));
plt.legend(['Bitcoin'], loc='upper left',fontsize = 'x-large');

###  Retrieve Data from Quandl's API
'''
Another possibility is to retrieve Bitcoin pricing data using Quandl's free [Bitcoin API](https://blog.quandl.com/api-for-bitcoin-data). For example, to obtain the daily bitcoin exchange rate (BTC vs. USD) on Bitstamp (Bitstamp is a bitcoin exchange based in Luxembourg) we use the code snippet below. The function `quandl_data` is inside the library `af`.
'''
quandl_id = 'BCHARTS/KRAKENUSD'
df_qdl = af.quandl_data(quandl_id)
df_qdl.columns = [c.lower().replace(' ', '_').replace('(', '').replace(')', '') for c in df_qdl.columns.values]

lst_col_2 = df_qdl.shape[1]-1
df_qdl.iloc[:,lst_col_2].plot(lw=3, figsize=(15,5));
plt.legend(['krakenUSD'], loc='upper left',fontsize = 'x-large');

### Retrieve Data from cryptocompare.com
'''
Another possibility is to retrieve data from [cryptocompare](https://www.cryptocompare.com/). In this case, we use the `requests` packages to make a `.get` request (the object `res` is a `Response` object) such as:

    res = requests.get(URL)
'''

res = requests.get('https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=USD&limit=2000')
df_cc = pd.DataFrame(json.loads(res.content)['Data']).set_index('time')
df_cc.index = pd.to_datetime(df_cc.index, unit='s')

lst_col = df_cc.shape[1]-1
df_cc.iloc[:,lst_col].plot(lw=2, figsize=(15,5));
plt.legend(['daily BTC'], loc='upper left',fontsize = 'x-large');

## Data Handling
'''
Let us start with `df` (from *bitFlyer*) from Kaggle.
'''

### Checking for `NaNs` and  making column titles cleaner

'''
Using `df.isnull().any()` we quickly see that the data does not contain null values. We get rid of upper cases, spaces, parentheses and so on, in the column titles.
'''


df.columns = [c.lower().replace(' ', '_').replace('(', '').replace(')', '') for c in df.columns.values]


### Group data by day and take the mean value

'''
Though Timestamps are in Unix time, this is easily taken care of using Python's `datetime` library:
- `pd.to_datetime` converts the argument to `datetime` 
- `Series.dt.date` gives a numpy array of Python `datetime.date` objects

We then drop the `Timestamp` column. The repeated dates occur simply because the data is collected minute-to-minute. Taking the daily mean using the method `daily_weighted_prices` from `af` we obtain:
'''
daily = af.daily_weighted_prices(df,'date','timestamp','s')


## Exploratory Data Analysis
'''
We can look for trends and seasonality in the data. Joining train and test sets and using the functions `trend`, `seasonal`, `residue` and `plot_components` for `af`:
'''
### Trend, seasonality and residue

dataset = daily.reset_index()
dataset['date'] = pd.to_datetime(dataset['date'])
dataset = dataset.set_index('date')

af.plot_comp(af.trend(dataset,'weighted_price'),
          af.seasonal(dataset,'weighted_price'),
          af.residue(dataset,'weighted_price'),
          af.actual(dataset,'weighted_price'))              

# Partial Auto-correlation (PACF)
'''
The PACF is the correlation (of the variable with itself) at a given lag, **controlling for the effect of previous (shorter) lags**. We see below that according to the PACF, prices with **one day** difference are highly correlated. After that the partial auto-correlation essentially drops to zero.
'''
plt.figure(figsize=(15,7))
ax = plt.subplot(211)
sm.graphics.tsa.plot_pacf(dataset['weighted_price'].values.squeeze(), lags=48, ax=ax)
plt.show();

## Train/Test split
'''
We now need to split our dataset. We need to fit the model using the training data and test the model with the test data. We can proceed as follows. We must choose the proportion of rows that will constitute the training set.
'''
train_size = 0.75
training_rows = train_size*len(daily)
int(training_rows)
'''
Then we slice `daily` using, for obvious reasons:

        [:int(training_rows)]
        [int(training_rows):]
        
We then have:
'''
train = daily[0:int(training_rows)]  
test = daily[int(training_rows):] 
print('Shapes of training and testing sets:',train.shape[0],'and',test.shape[0])
'''
We can automatize the split using a simple function for the library `aux_func`:
'''
test_size = 1 - train_size
train = af.train_test_split(daily, test_size=test_size)[0]
test = af.train_test_split(daily, test_size=test_size)[1]
print('Shapes of training and testing sets:',train.shape[0],'and',test.shape[0])
af.vc_to_df(train).head() 
af.vc_to_df(train).tail() 
af.vc_to_df(test).head()
af.vc_to_df(test).tail()
train.shape,test.shape

### Checking

af.vc_to_df(daily[0:199]).head() 
af.vc_to_df(daily[0:199]).tail() 
af.vc_to_df(daily[199:]).head()
af.vc_to_df(daily[199:]).tail()

### Reshaping
'''
We must reshape `train` and `test` for `Keras`.
'''

train = np.reshape(train, (len(train), 1));
test = np.reshape(test, (len(test), 1));

train.shape
test.shape

### `MinMaxScaler ` 
'''
We must now use `MinMaxScaler` which scales and translates features to a given range (between zero and one). 
'''

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

Reshaping once more:

X_train, Y_train = af.lb(train, 1)
X_test, Y_test = af.lb(test, 1)
X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))

'''
Now the shape is (number of examples, time steps, features per step).
'''

X_train.shape
X_test.shape

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(256))
model.add(Dense(1))

# compile and fit the model
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, Y_train, epochs=100, batch_size=32, 
                    validation_data=(X_test, Y_test))

model.summary()

## Train and test loss

### While the model is being trained, the train and test losses vary as shown in the figure below. The package `plotly.graph_objs` is extremely useful. The function `t( )` inside the argument is defined in `aux_func`:

py.iplot(dict(data=[af.t('loss','training_loss',history), af.t('val_loss','val_loss',history)], 
              layout=dict(title = 'history of training loss', xaxis = dict(title = 'epochs'),
                          yaxis = dict(title = 'loss'))), filename='training_process')

### Root-mean-square deviation (RMSE)

### The RMSE measures differences between values predicted by a model the actual values.

X_test_new = X_test.copy()
X_test_new = np.append(X_test_new, scaler.transform(dataset.iloc[-1][0]))
X_test_new = np.reshape(X_test_new, (len(X_test_new), 1, 1))
prediction = model.predict(X_test_new)

Inverting original scaling:

pred_inv = scaler.inverse_transform(prediction.reshape(-1, 1))
Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))
pred_inv_new = np.array(pred_inv[:,0][1:])
Y_test_new_inv = np.array(Y_test_inv[:,0])

### Renaming arrays for clarity

y_testing = Y_test_new_inv
y_predict = pred_inv_new

## Prediction versus True Values

layout = dict(title = 'True prices vs predicted prices',
             xaxis = dict(title = 'Day'), yaxis = dict(title = 'USD'))
fig = dict(data=[af.prediction_vs_true(y_testing,'Prediction'),
                 af.prediction_vs_true(y_predict,'True')],
           layout=layout)
py.iplot(fig, filename='results')

print('Prediction:\n')
print(list(y_predict[0:10]))
print('')
print('Test set:\n')
y_testing = [round(i,1) for i in list(Y_test_new_inv)]
print(y_testing[0:10])
print('')
print('Difference:\n')
diff = [round(abs((y_testing[i+1]-list(y_predict)[i])/list(y_predict)[i]),2) for i in range(len(y_predict)-1)]
print(diff[0:30])
print('')
print('Mean difference:\n')
print(100*round(np.mean(diff[0:30]),3),'%')

### The average difference is ~5%. There is something wrong here!

df = pd.DataFrame(data={'prediction':  y_predict.tolist(), 'testing': y_testing})

pct_variation = df.pct_change()[1:]
pct_variation = pct_variation[1:]

pct_variation.head()

layout = dict(title = 'True prices vs predicted prices variation (%)',
             xaxis = dict(title = 'Day'), yaxis = dict(title = 'USD'))
fig = dict(data=[af.prediction_vs_true(pct_variation['prediction'],'Prediction'),af.prediction_vs_true(pct_variation['testing'],'True')],
           layout=layout)
py.iplot(fig, filename='results')

## Altcoins

### Using the Poloniex API and two auxiliar function ([Ref.1](https://blog.patricktriest.com/analyzing-cryptocurrencies-python/)). Choosing the value of the end date to be today we have:

poloniex = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
start = datetime.strptime('2015-01-01', '%Y-%m-%d') # get data from the start of 2015
end = datetime.now() 
period = 86400 # day in seconds

def get_crypto_data(poloniex_pair):
    data_df = af.get_json_data(poloniex.format(poloniex_pair, 
                                               start.timestamp(),
                                               end.timestamp(), 
                                               period),
                            poloniex_pair)
    data_df = data_df.set_index('date')
    return data_df

lst_ac = ['ETH','LTC','XRP','ETC','STR','DASH','SC','XMR','XEM']
len(lst_ac)
ac_data = {}
for a in lst_ac:
    ac_data[a] = get_crypto_data('BTC_{}'.format(a))

lst_df = []
for el in lst_ac:
    print('Altcoin:',el)
    ac_data[el].head()
    lst_df.append(ac_data[el])

lst_df[0].head()
