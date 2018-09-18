# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:45:08 2018

@author: sshekhar
"""
import warnings
from stock_history import *
from datetime import datetime 
import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from math import sqrt, exp, log
import scipy.stats as scs
from arch import arch_model
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

import seaborn as sns
sns.set(style="darkgrid")

from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY, date2num
from mpl_finance import candlestick_ohlc
 
def pandas_candlestick_ohlc(dat, stick = "day", otherseries = None, title=None):
    """
    :param dat: pandas DataFrame object with datetime64 index, and float columns "Open", "High", "Low", and "Close", likely created via DataReader from "yahoo"
    :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines
 
    This will show a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
    """
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    dayFormatter = DateFormatter('%d')      # e.g., 12
 
    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    transdat = dat.loc[:,["Open", "High", "Low", "Close"]]
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1 # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years
            grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                            "High": max(group.High),
                                            "Low": min(group.Low),
                                            "Close": group.iloc[-1,3]},
                                           index = [group.index[0]]))
            if stick == "week": stick = 5
            elif stick == "month": stick = 30
            elif stick == "year": stick = 365
 
    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                        "High": max(group.High),
                                        "Low": min(group.Low),
                                        "Close": group.iloc[-1,3]},
                                       index = [group.index[0]]))
 
    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')
 
 
    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)
 
    ax.grid(True)
 
    # Create the candelstick chart
    candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                      plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                      colorup = "black", colordown = "red", width = stick * .4)
 
    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)
 
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    fig.suptitle(title)
    plt.show()

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 

start = datetime(2017,9,1)
end = datetime(2018,9,10)
tickersofinterest=["INFY","WIPRO","HCLTECH","TCS","TECHM"]
data=pd.DataFrame()
for sym in tickersofinterest:
    getdata = get_history(symbol=sym,
                    start=start, 
                    end=end)
    data=pd.concat([data,getdata])
#Confirm if we got all that we needed
data['Date']=pd.to_datetime(data['Date'])
print list(data), len(data), data['Symbol'].unique(), data.head()
infy_data=data[data.Symbol=="INFY"]
print infy_data.describe()



plt.figure(1) 
for sym in tickersofinterest:
    graphdata=data[data.Symbol==sym]
    print graphdata.describe()
    print graphdata.corr()
    print graphdata.cov()
    graphtickdata=graphdata[['Date','Close', 'Volume']].copy(deep=True)
    print graphtickdata.cov()
    graphtickdata.reset_index(drop=True, inplace=True)
    graphtickdata['Date']=pd.to_datetime(graphtickdata['Date'])
    graphtickdata.set_index(graphtickdata['Date'], inplace=True)
    y=graphtickdata['Close'].resample('W').mean().dropna()
    y.plot(figsize=(15, 6), grid=True, label=sym)
plt.legend()
plt.show()

tcssplitdate=datetime(2018,6,2)
infysplitdate=datetime(2018,9,4)
#Lets adjust TCS and INFY pre split price and volume data
plt.figure(2)
for sym in tickersofinterest:
    graphdata=data[data.Symbol==sym]
    if sym == "TCS":
        mask = (graphdata['Date'] < tcssplitdate)
        filterdata=graphdata.loc[mask]
        filterdata['Volume']=filterdata['Volume']*2
        filterdata['Close']=filterdata['Close']*0.5
        graphdata.loc[graphdata.Date.isin(filterdata['Date']),['Close','Volume']]=filterdata[['Close','Volume']]
    if sym == "INFY":        
        mask = (graphdata['Date'] < infysplitdate)
        filterdata=graphdata.loc[mask]
        filterdata['Volume']=filterdata['Volume']*(1.5)
        filterdata['Close']=filterdata['Close']*(0.66)
        graphdata.loc[graphdata.Date.isin(filterdata['Date']),['Close','Volume']]=filterdata[['Close','Volume']]
        infy_data.loc[infy_data.Date.isin(filterdata['Date']),['Close','Volume']]=filterdata[['Close','Volume']]
    graphtickdata=graphdata[['Date','Close']].copy(deep=True)  
    graphtickdata.reset_index(drop=True, inplace=True)
    graphtickdata['Date']=pd.to_datetime(graphtickdata['Date'])
    graphtickdata.set_index(graphtickdata['Date'], inplace=True)

    y=graphtickdata['Close'].resample('W').mean().dropna()
    y.plot(figsize=(15, 6), grid=True, label=sym)
plt.legend()
plt.show()

plot_acf(graphdata['Close'])
plt.show()
plot_pacf(graphdata['Close'], lags=5)
plt.show()
    
for sym in tickersofinterest:
    candledata=data[data.Symbol==sym]
    candlestickdata=candledata[['Date','Open', 'High', 'Low','Close']].copy(deep=True)
    if sym == "TCS":
        mask = (candlestickdata['Date'] < tcssplitdate)
        filterdata=candlestickdata.loc[mask]
        filterdata['Close']=filterdata['Close']*0.5
        filterdata['Open']=filterdata['Open']*0.5
        filterdata['High']=filterdata['High']*0.5
        filterdata['Low']=filterdata['Low']*0.5
        candlestickdata.loc[candlestickdata.Date.isin(filterdata['Date']),['Close','Open', 'High', 'Low']]=filterdata[['Close','Open', 'High', 'Low']]
    if sym == "INFY":
        mask = (candlestickdata['Date'] < infysplitdate)
        filterdata=candlestickdata.loc[mask]
        filterdata['Close']=filterdata['Close']*0.66
        filterdata['Open']=filterdata['Open']*0.66
        filterdata['High']=filterdata['High']*0.66
        filterdata['Low']=filterdata['Low']*0.66
        candlestickdata.loc[candlestickdata.Date.isin(filterdata['Date']),['Close','Open', 'High', 'Low']]=filterdata[['Close','Open', 'High', 'Low']]
        infy_data.loc[infy_data.Date.isin(filterdata['Date']),['Close','Open', 'High', 'Low']]=filterdata[['Close','Open', 'High', 'Low']]
    candlestickdata.reset_index(drop=True, inplace=True)
    candlestickdata['Date']=pd.to_datetime(candlestickdata['Date'])
    candlestickdata.set_index(candlestickdata['Date'], inplace=True)
    candlestickdata=candlestickdata[-15:]
    pandas_candlestick_ohlc(candlestickdata,title=sym)


# From the loop above, we know that graphdata will contain adjusted values of Infy
infy_data['Date']=pd.to_datetime(infy_data['Date'], infer_datetime_format=True)
infy_data.set_index(infy_data['Date'], inplace=True)
tsplot(infy_data['Close'], lags=5)

tsplot(np.diff(infy_data['Close']), lags=30)

plt.figure("Auto Regression")
#Add NaN values for all dates, so that the regressor gets a daily freq
#Extend infy_data to have all dates.
#infy_data.reset_index(drop=True, inplace=True)

ix = pd.DatetimeIndex(start=start, end=end, freq='D', dtype='datetime64[ns]', name='fillDate')
infy_data = infy_data.reindex(ix, fill_value=0)
infy_data['Symbol']="INFY"

i=0
while i < len(infy_data):
    if infy_data.ix[i,'Close']==0.00 and i != 0:
        infy_data.ix[i,'Close']=infy_data.ix[i-1,'Close']
        print infy_data.iloc[[i-1],8]
    i+=1

X = infy_data['Close']
# split dataset into train and test. We will retain last 7 observations to test our model
train, test = X[1:len(X)-14], X[len(X)-14:]
# train autoregression
model = smt.AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('\nCoefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
test.plot(figsize=(15, 6), grid=True, label="Test")
predictions.plot(color='orange', label="Predictions")
plt.legend()
plt.show()


# create a differe
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return Series(diff)

# difference data to confirm if difference data is stationary.
series = Series(X)
stationary = difference(X)
stationary.index = series.index[1:]
# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
#Check ACF and PACF in a single chart
series = Series(stationary)
plt.figure()
plt.subplot(211)
plot_acf(series, ax=pyplot.gca())
plt.subplot(212)
plot_pacf(series, ax=pyplot.gca())
plt.show()



# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) * 0.04)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# load dataset
series = Series(X)
# evaluate parameters
p_values = range(0, 4)
d_values = range(0, 4)
q_values = range(0, 8)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)

#Process the best ARIMA model 
# prepare data
Y = X.astype('float32')
train, test = Y[1:len(Y)-14], Y[len(Y)-14:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	model = ARIMA(history, order=(0,1,0))
	model_fit = model.fit(disp=0)
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = pd.DataFrame(residuals)
plt.figure("Residual Errors")
plt.subplot(211)
residuals.hist(ax=pyplot.gca())
plt.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
plt.show()

#Check ACF and PACF for residual error to verify any corelation
plt.figure()
plt.subplot(211)
plot_acf(residuals, ax=pyplot.gca())
plt.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
plt.show()

#Residual errors are high and difficult to explain. But for completeness sake, we will save the model and use it for prediction before moving to GARCH.

# transform data
transformed, lam = boxcox(X)
# fit model
model = ARIMA(transformed, order=(0,1,0))
model_fit = model.fit(disp=0)
# save model
model_fit.save('model.pkl')
np.save('model_lambda.npy', [lam])

#Predict by restoring the model and lambda values and then use forecast to forecast the next date value
# invert box-cox transform
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)
# load model 
model_fit = ARIMAResults.load('model.pkl')
lam = np.load('model_lambda.npy')
yhat = model_fit.forecast()[0]
yhat = boxcox_inverse(yhat, lam)
print('Predicted: %.3f' % yhat)

#Lets validate the results of ARIMA model prediction for the entire test sample

 
# load the train and test data and prepare datasets
dataset = Series(train)
X = dataset.values.astype('float32')
history = [x for x in X]
validation = Series(test)
y = validation.values.astype('float32')
# make first prediction
predictions = list()
yhat = model_fit.forecast()[0]
yhat = boxcox_inverse(yhat, lam)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
	# transform
	transformed, lam = boxcox(history)
	if lam < -5:
		transformed, lam = history, 1
	# predict
	model = ARIMA(transformed, order=(0,1,0))
	model_fit = model.fit(disp=0)
	yhat = model_fit.forecast()[0]
	# invert transformed prediction
	yhat = boxcox_inverse(yhat, lam)
	predictions.append(yhat)
	# observation
	obs = y[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(y, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
plt.figure("Prediction Vs Actual INFY Closing Price")
plt.plot(y, label="Actual", color='blue')
plt.plot(predictions,label="Predicted", color='red')
plt.legend()
plt.show()
