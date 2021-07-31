#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 11:25:49 2021

@author: oodaye
"""

#%% use FB prophet to determine how to predict a time series 
#%% get the initial imports 
# Initial imports
import os
from datetime import datetime 
import requests
import pandas as pd
import json

from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()

from fbprophet import Prophet

import matplotlib as plt
#%matplotlib inline

#%% set up the directory
os.chdir("/home/oodaye/Fintech/class_repo/python-homework-repo/python-homework/Project_1")

#%% load the data set 
dt = pd.read_csv("btc-usd-max.csv")
dt = dt.dropna()

dt1 = pd.DataFrame( { 'ds' : dt['snapped_at'] , 'y' : dt['price']})

## remove time zone and replace the date time column 
q = dt1['ds']
q = pd.to_datetime(q)
q = q.dt.tz_convert(None)

dt1['ds'] = q

# send the data to FB Prophet 

m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(dt1) # fit the model using all data

#  plot the data 
future = m.make_future_dataframe(periods=90) #we need to specify the number of days in future
prediction = m.predict(future)
m.plot(prediction)
plt.pyplot.title("Prediction of Bitcoin Price")
plt.pyplot.xlabel("Date")
plt.pyplot.ylabel("Close Price")
plt.pyplot.savefig("btc_forecast_full.png")
#plt.show()

#%% use the most recent 300 days for bitcoin
dt2 = dt1.tail(300)

m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(dt2) # fit the model using all data

#  plot the data 
future = m.make_future_dataframe(periods=90) #we need to specify the number of days in future
prediction = m.predict(future)
m.plot(prediction)
plt.pyplot.title("Prediction of Bitcoin Price -- Last 300 Days")
plt.pyplot.xlabel("Date")
plt.pyplot.ylabel("Close Price")
plt.pyplot.savefig("btc_forecast_full_300days.png")

#%% plot ehtereum forecast 

dt = pd.read_csv("eth-usd-max.csv")
dt = dt.dropna()

dt3 = pd.DataFrame( { 'ds' : dt['snapped_at'] , 'y' : dt['price']})

## remove time zone and replace the date time column 
q = dt3['ds']
q = pd.to_datetime(q)
q = q.dt.tz_convert(None)

dt3['ds'] = q

# send the data to FB Prophet 

m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(dt3) # fit the model using all data

#  plot the data 
future = m.make_future_dataframe(periods=90) #we need to specify the number of days in future
prediction = m.predict(future)
m.plot(prediction)
plt.pyplot.title("Prediction of Ethereum Price")
plt.pyplot.xlabel("Date")
plt.pyplot.ylabel("Close Price")
plt.pyplot.savefig("eth_forecast_full.png")

#%% use the most recent 100 days for ethereum 
dt4 = dt3.tail(400)

m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(dt4) # fit the model using all data

#  plot the data 
future = m.make_future_dataframe(periods=90) #we need to specify the number of days in future
prediction = m.predict(future)
m.plot(prediction)
plt.pyplot.title("Prediction of Ethereum Price -- Last 400 Days")
plt.pyplot.xlabel("Date")
plt.pyplot.ylabel("Close Price")
plt.pyplot.savefig("eth_forecast_full_400days.png")

#%% plot DAI forecast 

dt = pd.read_csv("dai-usd-max.csv")
dt = dt.dropna()

dt5 = pd.DataFrame( { 'ds' : dt['snapped_at'] , 'y' : dt['price']})

## remove time zone and replace the date time column 
q = dt5['ds']
q = pd.to_datetime(q)
q = q.dt.tz_convert(None)

dt5['ds'] = q

# send the data to FB Prophet 

m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(dt5) # fit the model using all data

#  plot the data 
future = m.make_future_dataframe(periods=90) #we need to specify the number of days in future
prediction = m.predict(future)
m.plot(prediction)
plt.pyplot.title("Prediction of DAI Price")
plt.pyplot.xlabel("Date")
plt.pyplot.ylabel("Close Price")
plt.pyplot.savefig("dai_forecast_full.png")