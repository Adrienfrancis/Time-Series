# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 18:22:20 2018

@author: Adrien
"""

# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy as np
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas as pd
print('pandas: %s' % pandas.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)

from pandas import Series


###################################################################CPI
#Split into dataset and validation
CPI_Data = Series.from_csv('Og_CPI.csv', header=0)
CPI_Data
CPI_split_point = len(CPI_Data) - 6
CPI_dataset, CPI_validation = CPI_Data[0:CPI_split_point], CPI_Data[CPI_split_point:]
print('Dataset %d, Validation %d' % (len(CPI_dataset), len(CPI_validation)))
CPI_dataset.to_csv('dataset.csv')
CPI_validation.to_csv('validation.csv')

############# SUMMARY STATISTICS

CPI_series = Series.from_csv('dataset.csv')
print(CPI_series.describe())

#line plot
from matplotlib import pyplot
CPI_series.plot()
pyplot.show()

#histogram
pyplot.hist(CPI_series)
pyplot.show()

#seasonal line plots
from pandas import DataFrame
from pandas import TimeGrouper
CPI_groups = CPI_series['2012':'2017'].groupby(TimeGrouper('A'))
CPI_years = DataFrame()
pyplot.figure()
i = 1
n_CPI_groups = len(CPI_groups)
for name, group in CPI_groups:
	pyplot.subplot((n_CPI_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()

#density plot

pyplot.figure(1)
pyplot.subplot(211)
CPI_series.hist()
pyplot.subplot(212)
CPI_series.plot(kind='kde')
pyplot.show()

#box and whisker plot

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
CPI_series = Series.from_csv('dataset.csv')
CPI_groups = CPI_series['2012':'2017'].groupby(TimeGrouper('A'))
CPI_years = DataFrame()
for name, group in CPI_groups:
	CPI_years[name.year] = group.values
CPI_years.boxplot()
pyplot.show()

from statsmodels.tsa.seasonal import seasonal_decompose
CPI_result = seasonal_decompose(CPI_series, model='multiplicative')
print(CPI_result.trend)
print(CPI_result.seasonal)
print(CPI_result.resid)
print(CPI_result.observed)
CPI_result.plot()
pyplot.show()

############## NAIVE FORECAST PERFORMANCE
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
CPI_series = Series.from_csv('dataset.csv')
# prepare data
CPI_X = CPI_series.values
CPI_X = CPI_X.astype('float32')
CPI_train_size = int(len(CPI_X) * 0.50)
CPI_train, CPI_test = CPI_X[0:CPI_train_size], CPI_X[CPI_train_size:]
# walk-forward validation for past 9 values
CPI_history = [x for x in CPI_train]
CPI_predictions = list()
for i in range(len(CPI_test)):
	# predict
	yhat = CPI_history[-6]
	CPI_predictions.append(yhat)
	# observation
	obs = CPI_test[i]
	CPI_history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
mse = mean_squared_error(CPI_test, CPI_predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)


############ARIMA for seasonal data

from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
 
# create a differenced CPI_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(CPI_history, yhat, interval=1):
	return yhat + CPI_history[-interval]

# Double difference data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
CPI_months_in_year = 1
stationary = difference(CPI_X, CPI_months_in_year)
stationary.index = CPI_series.index[CPI_months_in_year:]
CPI_months_in_year = 1
stationary = difference(stationary, CPI_months_in_year)
stationary.index = CPI_series.index[CPI_months_in_year+1:]
stationary

# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
# plot
stationary.plot()
pyplot.show()

#ACF AND PACF PLOTS

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
CPI_series = Series.from_csv('stationary.csv')
pyplot.figure()
pyplot.subplot(211)
plot_acf(CPI_series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(CPI_series, ax=pyplot.gca())
pyplot.show()

#Arima
# create a differenced CPI_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(CPI_history, yhat, interval=1):
	return yhat + CPI_history[-interval]


# load data
CPI_series = Series.from_csv('dataset.csv')
# prepare data
CPI_X = CPI_series.values
CPI_X = CPI_X.astype('float32')
CPI_train_size = int(len(CPI_X) * 0.50)
CPI_train, CPI_test = CPI_X[0:CPI_train_size], CPI_X[CPI_train_size:]

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
# walk-forward validation(re-run from here)
CPI_history = [x for x in CPI_train]
CPI_predictions = list()
for i in range(len(CPI_test)):
    # difference data
    CPI_months_in_year = 1
    diff = difference(CPI_history, CPI_months_in_year)
    diff2 = diff
    diff = difference(diff2, CPI_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit( trend='nc',disp=0,)
    yhat = model_fit.forecast()[0]
    yhat = inverse_difference(diff2, yhat, CPI_months_in_year)
    yhat = inverse_difference(CPI_history, yhat, CPI_months_in_year) 
    CPI_predictions.append(yhat)
    # observation
    obs = yhat
    CPI_history.append(obs)
# report performance
mse = mean_squared_error(CPI_test, CPI_predictions)
rmse = sqrt(mse)
print('RMSE: %.4f' % rmse)

# errors
residuals = [CPI_test[i]-CPI_predictions[i] for i in range(len(CPI_test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()

############################################################# Prediction
 
#Getting data
import datetime
from dateutil.relativedelta import relativedelta
Data = Series.from_csv('Og_CPI.csv', header=0)
CPI_New_Data=Data
last_data_date = Data.iloc[-1:].index[0]
print ('Last data was collected ', last_data_date)
last_data_date.date()
first_data_date = pd.Timestamp(last_data_date.date()+ relativedelta(months=1))
periodicity = 'm'
dfFuture = pd.DataFrame({ 'day': pd.Series([first_data_date, datetime.datetime(2019,7,31)])})
dfFuture.set_index('day',inplace=True)
dfFuture = dfFuture.asfreq(periodicity)
n=len(dfFuture)
CPI_New_Data = CPI_New_Data.append(dfFuture)
CPI_New_Data.columns =['CPI']
CPI_New_Data


#splitting data n= number of nan values
split_point = len(CPI_New_Data) - n
dataset, validation = CPI_New_Data[0:split_point], CPI_New_Data[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# load and prepare datasets
dataset = Series.from_csv('dataset.csv',header=0)
CPI_X = dataset.values.astype('float32')
CPI_history = [x for x in CPI_X]
validation = Series.from_csv('validation.csv',header=0)
CPI_y = validation.values.astype('float32')

# create a differenced CPI_series function
def difference(dataset, interval=1):
	diff = list()
	for i in range(1, len(dataset)):
		value = (dataset[i] - dataset[i - interval])
		diff.append(value)
	return diff
 
# invert differenced value funtion
def inverse_difference(CPI_history, yhat, interval=1):
	return yhat + CPI_history[-interval]
 
# make first prediction
CPI_predictions = list()
CPI_bias = 0
# rolling forecasts
for i in range(0, len(CPI_y)):
    # difference data
    CPI_months_in_year = 1
    diff = difference(CPI_history, CPI_months_in_year)
    diff2 = diff
    diff = difference(diff2, CPI_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    yhat = inverse_difference(diff2, yhat, CPI_months_in_year)
    yhat = CPI_bias+inverse_difference(CPI_history, yhat, CPI_months_in_year)
    CPI_predictions.append(yhat)
    # observation
    obs = yhat
    CPI_history.append(obs)

CPI_predictions
CPI_New_Data
for i in range(0,n):
    CPI_New_Data.iloc[-n+i:] = CPI_predictions[i]
    
CPI_New_Data.index.name='Date'
Pred_CPI=CPI_New_Data
CPI_New_Data

################################################################################################Income
#Split into dataset and validation
Income_Data = Series.from_csv('Og_Income.csv', header=0)
Income_Data
Income_split_point = len(Income_Data) - 5
Income_dataset, Income_validation = Income_Data[0:Income_split_point], Income_Data[Income_split_point:]
print('Dataset %d, Validation %d' % (len(Income_dataset), len(Income_validation)))
Income_dataset.to_csv('dataset.csv')
Income_validation.to_csv('validation.csv')

############# SUMMARY STATISTICS

Income_series = Series.from_csv('dataset.csv')
print(Income_series.describe())

#line plot
from matplotlib import pyplot
Income_series.plot()
pyplot.show()

#histogram
pyplot.hist(Income_series)
pyplot.show()

#seasonal line plots
from pandas import DataFrame
from pandas import TimeGrouper
Income_groups = Income_series['2012':'2017'].groupby(TimeGrouper('A'))
Income_years = DataFrame()
pyplot.figure()
i = 1
n_Income_groups = len(Income_groups)
for name, group in Income_groups:
	pyplot.subplot((n_Income_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()

#density plot

pyplot.figure(1)
pyplot.subplot(211)
Income_series.hist()
pyplot.subplot(212)
Income_series.plot(kind='kde')
pyplot.show()

#box and whisker plot

"""from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
Income_series = Series.from_csv('dataset.csv')
Income_groups = Income_series['2012':'2017'].groupby(TimeGrouper('A'))
Income_years = DataFrame()
for name, group in Income_groups:
	Income_years[name.year] = group.values
Income_years.boxplot()
pyplot.show()
"""
from statsmodels.tsa.seasonal import seasonal_decompose
Income_result = seasonal_decompose(Income_series, model='multiplicative')
print(Income_result.trend)
print(Income_result.seasonal)
print(Income_result.resid)
print(Income_result.observed)
Income_result.plot()
pyplot.show()

############## NAIVE FORECAST PERFORMANCE
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
Income_series = Series.from_csv('dataset.csv')
# prepare data
Income_X = Income_series.values
Income_X = Income_X.astype('float32')
Income_train_size = int(len(Income_X) * 0.50)
Income_train, Income_test = Income_X[0:Income_train_size], Income_X[Income_train_size:]
# walk-forward validation for past 9 values
Income_history = [x for x in Income_train]
Income_predictions = list()
for i in range(len(Income_test)):
	# predict
	yhat = Income_history[-6]
	Income_predictions.append(yhat)
	# observation
	obs = Income_test[i]
	Income_history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
mse = mean_squared_error(Income_test, Income_predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)


############ARIMA for seasonal data

from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
 
# create a differenced Income_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(Income_history, yhat, interval=1):
	return yhat + Income_history[-interval]

#difference data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
Income_months_in_year = 12
stationary = difference(Income_X, Income_months_in_year)
stationary.index = Income_series.index[Income_months_in_year:]

#Income_months_in_year = 1
#stationary = difference(stationary, Income_months_in_year)
#stationary.index = Income_series.index[Income_months_in_year+1:]
#stationary

# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
# plot
stationary.plot()
pyplot.show()

#ACF AND PACF PLOTS

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
Income_series = Series.from_csv('stationary.csv')
pyplot.figure()
pyplot.subplot(211)
plot_acf(Income_series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(Income_series, ax=pyplot.gca())
pyplot.show()

#Arima
# create a differenced Income_series
def difference(dataset, interval):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(Income_history, yhat, interval):
	return yhat + Income_history[-interval]


# load data
Income_series = Series.from_csv('dataset.csv')
# prepare data
Income_X = Income_series.values
Income_X = Income_X.astype('float32')
Income_train_size = int(len(Income_X) * 0.50)
Income_train, Income_test = Income_X[0:Income_train_size], Income_X[Income_train_size:]

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

# walk-forward validation(re-run from here)
Income_history = [x for x in Income_train]
Income_predictions = list()
for i in range(len(Income_test)):
    # difference data
    Income_months_in_year = 12
    diff = difference(Income_history, Income_months_in_year)
    #diff2 = diff
    #diff = difference(diff2, Income_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit( trend='nc',disp=0,)
    yhat = model_fit.forecast()[0]
    #yhat = inverse_difference(diff2, yhat, Income_months_in_year)
    yhat = inverse_difference(Income_history, yhat, Income_months_in_year) 
    Income_predictions.append(yhat)
    # observation
    obs = yhat
    Income_history.append(obs)
# report performance
mse = mean_squared_error(Income_test, Income_predictions)
rmse = sqrt(mse)
print('RMSE: %.4f' % rmse)

# errors
residuals = [Income_test[i]-Income_predictions[i] for i in range(len(Income_test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()

############################################################# Prediction
 
#Getting data
import datetime
from dateutil.relativedelta import relativedelta
Data = Series.from_csv('Og_Income.csv', header=0)
Income_New_Data=Data
last_data_date = Data.iloc[-1:].index[0]
print ('Last data was collected ', last_data_date)
last_data_date.date()
first_data_date = pd.Timestamp(last_data_date.date()+ relativedelta(months=1))
periodicity = 'm'
dfFuture = pd.DataFrame({ 'day': pd.Series([first_data_date, datetime.datetime(2019,7,31)])})
dfFuture.set_index('day',inplace=True)
dfFuture = dfFuture.asfreq(periodicity)
n=len(dfFuture)
Income_New_Data = Income_New_Data.append(dfFuture)
Income_New_Data.columns =['Income']
Income_New_Data


#splitting data n= number of nan values
split_point = len(Income_New_Data) - n
dataset, validation = Income_New_Data[0:split_point], Income_New_Data[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# load and prepare datasets
dataset = Series.from_csv('dataset.csv',header=0)
Income_X = dataset.values.astype('float32')
Income_history = [x for x in Income_X]
validation = Series.from_csv('validation.csv',header=0)
Income_y = validation.values.astype('float32')

# create a differenced Income_series function
def difference(dataset, interval):
	diff = list()
	for i in range(1, len(dataset)):
		value = (dataset[i] - dataset[i - interval])
		diff.append(value)
	return diff
 
# invert differenced value funtion
def inverse_difference(Income_history, yhat, interval=1):
	return yhat + Income_history[-interval]
 
# make first prediction
Income_predictions = list()
Income_bias = 0
# rolling forecasts
for i in range(0, len(Income_y)):
    # difference data
    Income_months_in_year = 12
    diff = difference(Income_history, Income_months_in_year)
    #diff2 = diff
    #diff = difference(diff2, Income_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    #yhat = inverse_difference(diff2, yhat, Income_months_in_year)
    yhat = Income_bias+inverse_difference(Income_history, yhat, Income_months_in_year)
    Income_predictions.append(yhat)
    # observation
    obs = yhat
    Income_history.append(obs)

Income_predictions
Income_New_Data
for i in range(0,n):
    Income_New_Data.iloc[-n+i:] = Income_predictions[i]
    
Income_New_Data.index.name='Date'
Pred_Income=Income_New_Data
Income_New_Data

############################################################################Inflation
#Split into dataset and validation
Inflation_Data = Series.from_csv('Og_Inflation.csv', header=0)
Inflation_Data
Inflation_split_point = len(Inflation_Data) - 6
Inflation_dataset, Inflation_validation = Inflation_Data[0:Inflation_split_point], Inflation_Data[Inflation_split_point:]
print('Dataset %d, Validation %d' % (len(Inflation_dataset), len(Inflation_validation)))
Inflation_dataset.to_csv('dataset.csv')
Inflation_validation.to_csv('validation.csv')

############# SUMMARY STATISTICS

Inflation_series = Series.from_csv('dataset.csv')
print(Inflation_series.describe())

#line plot
from matplotlib import pyplot
Inflation_series.plot()
pyplot.show()

#histogram
pyplot.hist(Inflation_series)
pyplot.show()

#seasonal line plots
from pandas import DataFrame
from pandas import TimeGrouper
Inflation_groups = Inflation_series['2012':'2017'].groupby(TimeGrouper('A'))
Inflation_years = DataFrame()
pyplot.figure()
i = 1
n_Inflation_groups = len(Inflation_groups)
for name, group in Inflation_groups:
	pyplot.subplot((n_Inflation_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()

#density plot

pyplot.figure(1)
pyplot.subplot(211)
Inflation_series.hist()
pyplot.subplot(212)
Inflation_series.plot(kind='kde')
pyplot.show()

#box and whisker plot

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
Inflation_series = Series.from_csv('dataset.csv')
Inflation_groups = Inflation_series['2012':'2017'].groupby(TimeGrouper('A'))
Inflation_years = DataFrame()
for name, group in Inflation_groups:
	Inflation_years[name.year] = group.values
Inflation_years.boxplot()
pyplot.show()

"""from statsmodels.tsa.seasonal import seasonal_decompose
Inflation_result = seasonal_decompose(Inflation_series, model='multiplicative')
print(Inflation_result.trend)
print(Inflation_result.seasonal)
print(Inflation_result.resid)
print(Inflation_result.observed)
Inflation_result.plot()
pyplot.show()"""

############## NAIVE FORECAST PERFORMANCE
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
Inflation_series = Series.from_csv('dataset.csv')
# prepare data
Inflation_X = Inflation_series.values
Inflation_X = Inflation_X.astype('float32')
Inflation_train_size = int(len(Inflation_X) * 0.50)
Inflation_train, Inflation_test = Inflation_X[0:Inflation_train_size], Inflation_X[Inflation_train_size:]
# walk-forward validation for past 9 values
Inflation_history = [x for x in Inflation_train]
Inflation_predictions = list()
for i in range(len(Inflation_test)):
	# predict
	yhat = Inflation_history[-6]
	Inflation_predictions.append(yhat)
	# observation
	obs = Inflation_test[i]
	Inflation_history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
mse = mean_squared_error(Inflation_test, Inflation_predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)


############ARIMA for seasonal data

from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
 
# create a differenced Inflation_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(Inflation_history, yhat, interval=1):
	return yhat + Inflation_history[-interval]

# Double difference data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
Inflation_months_in_year = 1
stationary = difference(Inflation_X, Inflation_months_in_year)
stationary.index = Inflation_series.index[Inflation_months_in_year:]

"""Inflation_months_in_year = 1
stationary = difference(stationary, Inflation_months_in_year)
stationary.index = Inflation_series.index[Inflation_months_in_year+1:]
stationary"""

# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
# plot
stationary.plot()
pyplot.show()

#ACF AND PACF PLOTS

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
Inflation_series = Series.from_csv('stationary.csv')
pyplot.figure()
pyplot.subplot(211)
plot_acf(Inflation_series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(Inflation_series, ax=pyplot.gca())
pyplot.show()

#Arima
# create a differenced Inflation_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(Inflation_history, yhat, interval=1):
	return yhat + Inflation_history[-interval]


# load data
Inflation_series = Series.from_csv('dataset.csv')
# prepare data
Inflation_X = Inflation_series.values
Inflation_X = Inflation_X.astype('float32')
Inflation_train_size = int(len(Inflation_X) * 0.50)
Inflation_train, Inflation_test = Inflation_X[0:Inflation_train_size], Inflation_X[Inflation_train_size:]

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
# walk-forward validation(re-run from here)
Inflation_history = [x for x in Inflation_train]
Inflation_predictions = list()
for i in range(len(Inflation_test)):
    # difference data
    Inflation_months_in_year = 1
    diff = difference(Inflation_history, Inflation_months_in_year)
    #diff2 = diff
    #diff = difference(diff2, Inflation_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit( trend='nc',disp=0,)
    yhat = model_fit.forecast()[0]
    #yhat = inverse_difference(diff2, yhat, Inflation_months_in_year)
    yhat = inverse_difference(Inflation_history, yhat, Inflation_months_in_year) 
    Inflation_predictions.append(yhat)
    # observation
    obs = yhat
    Inflation_history.append(obs)
# report performance
mse = mean_squared_error(Inflation_test, Inflation_predictions)
rmse = sqrt(mse)
print('RMSE: %.4f' % rmse)

# errors
residuals = [Inflation_test[i]-Inflation_predictions[i] for i in range(len(Inflation_test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()

############################################################# Prediction
 
#Getting data
import datetime
from dateutil.relativedelta import relativedelta
Data = Series.from_csv('Og_Inflation.csv', header=0)
Inflation_New_Data=Data
last_data_date = Data.iloc[-1:].index[0]
print ('Last data was collected ', last_data_date)
last_data_date.date()
first_data_date = pd.Timestamp(last_data_date.date()+ relativedelta(months=1))
periodicity = 'm'
dfFuture = pd.DataFrame({ 'day': pd.Series([first_data_date, datetime.datetime(2019,7,31)])})
dfFuture.set_index('day',inplace=True)
dfFuture = dfFuture.asfreq(periodicity)
n=len(dfFuture)
Inflation_New_Data = Inflation_New_Data.append(dfFuture)
Inflation_New_Data.columns =['Inflation']
Inflation_New_Data


#splitting data n= number of nan values
split_point = len(Inflation_New_Data) - n
dataset, validation = Inflation_New_Data[0:split_point], Inflation_New_Data[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# load and prepare datasets
dataset = Series.from_csv('dataset.csv',header=0)
Inflation_X = dataset.values.astype('float32')
Inflation_history = [x for x in Inflation_X]
validation = Series.from_csv('validation.csv',header=0)
Inflation_y = validation.values.astype('float32')

# create a differenced Inflation_series function
def difference(dataset, interval=1):
	diff = list()
	for i in range(1, len(dataset)):
		value = (dataset[i] - dataset[i - 1])
		diff.append(value)
	return diff
 
# invert differenced value funtion
def inverse_difference(Inflation_history, yhat, interval=1):
	return yhat + Inflation_history[-interval]
 
# make first prediction
Inflation_predictions = list()
Inflation_bias = 0
# rolling forecasts
for i in range(0, len(Inflation_y)):
    # difference data
    Inflation_months_in_year = 1
    diff = difference(Inflation_history, Inflation_months_in_year)
    #diff2 = diff
    #diff = difference(diff2, Inflation_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    #yhat = inverse_difference(diff2, yhat, Inflation_months_in_year)
    yhat = Inflation_bias+inverse_difference(Inflation_history, yhat, Inflation_months_in_year)
    Inflation_predictions.append(yhat)
    # observation
    obs = yhat
    Inflation_history.append(obs)

Inflation_predictions
Inflation_New_Data
for i in range(0,n):
    Inflation_New_Data.iloc[-n+i:] = Inflation_predictions[i]
    
Inflation_New_Data.index.name='Date'
Pred_Inflation=Inflation_New_Data
Inflation_New_Data

####################################################################################################Inventory
#Split into dataset and validation
Inventory_Data = Series.from_csv('Og_Inventory.csv', header=0)
Inventory_Data
Inventory_split_point = len(Inventory_Data) - 5
Inventory_dataset, Inventory_validation = Inventory_Data[0:Inventory_split_point], Inventory_Data[Inventory_split_point:]
print('Dataset %d, Validation %d' % (len(Inventory_dataset), len(Inventory_validation)))
Inventory_dataset.to_csv('dataset.csv')
Inventory_validation.to_csv('validation.csv')

############# SUMMARY STATISTICS

Inventory_series = Series.from_csv('dataset.csv')
print(Inventory_series.describe())

#line plot
from matplotlib import pyplot
Inventory_series.plot()
pyplot.show()

#histogram
pyplot.hist(Inventory_series)
pyplot.show()

#seasonal line plots
from pandas import DataFrame
from pandas import TimeGrouper
Inventory_groups = Inventory_series['2012':'2017'].groupby(TimeGrouper('A'))
Inventory_years = DataFrame()
pyplot.figure()
i = 1
n_Inventory_groups = len(Inventory_groups)
for name, group in Inventory_groups:
	pyplot.subplot((n_Inventory_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()

#density plot

pyplot.figure(1)
pyplot.subplot(211)
Inventory_series.hist()
pyplot.subplot(212)
Inventory_series.plot(kind='kde')
pyplot.show()

#box and whisker plot

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
Inventory_series = Series.from_csv('dataset.csv')
Inventory_groups = Inventory_series['2012':'2017'].groupby(TimeGrouper('A'))
Inventory_years = DataFrame()
for name, group in Inventory_groups:
	Inventory_years[name.year] = group.values
Inventory_years.boxplot()
pyplot.show()

from statsmodels.tsa.seasonal import seasonal_decompose
Inventory_result = seasonal_decompose(Inventory_series, model='multiplicative')
print(Inventory_result.trend)
print(Inventory_result.seasonal)
print(Inventory_result.resid)
print(Inventory_result.observed)
Inventory_result.plot()
pyplot.show()

############## NAIVE FORECAST PERFORMANCE
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
Inventory_series = Series.from_csv('dataset.csv')
# prepare data
Inventory_X = Inventory_series.values
Inventory_X = Inventory_X.astype('float32')
Inventory_train_size = int(len(Inventory_X) * 0.50)
Inventory_train, Inventory_test = Inventory_X[0:Inventory_train_size], Inventory_X[Inventory_train_size:]
# walk-forward validation for past 9 values
Inventory_history = [x for x in Inventory_train]
Inventory_predictions = list()
for i in range(len(Inventory_test)):
	# predict
	yhat = Inventory_history[-6]
	Inventory_predictions.append(yhat)
	# observation
	obs = Inventory_test[i]
	Inventory_history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
mse = mean_squared_error(Inventory_test, Inventory_predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)


############ARIMA for seasonal data

from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
 
# create a differenced Inventory_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(Inventory_history, yhat, interval=1):
	return yhat + Inventory_history[-interval]

# Double difference data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
Inventory_months_in_year = 12
stationary = difference(Inventory_X, Inventory_months_in_year)
stationary.index = Inventory_series.index[Inventory_months_in_year:]

"""Inventory_months_in_year = 1
stationary = difference(stationary, Inventory_months_in_year)
stationary.index = Inventory_series.index[Inventory_months_in_year+1:]
stationary"""

# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
# plot
stationary.plot()
pyplot.show()

#ACF AND PACF PLOTS

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
Inventory_series = Series.from_csv('stationary.csv')
pyplot.figure()
pyplot.subplot(211)
plot_acf(Inventory_series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(Inventory_series, ax=pyplot.gca())
pyplot.show()

#Arima
# create a differenced Inventory_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(Inventory_history, yhat, interval=1):
	return yhat + Inventory_history[-interval]


# load data
Inventory_series = Series.from_csv('dataset.csv')
# prepare data
Inventory_X = Inventory_series.values
Inventory_X = Inventory_X.astype('float32')
Inventory_train_size = int(len(Inventory_X) * 0.50)
Inventory_train, Inventory_test = Inventory_X[0:Inventory_train_size], Inventory_X[Inventory_train_size:]

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
# walk-forward validation(re-run from here)
Inventory_history = [x for x in Inventory_train]
Inventory_predictions = list()
for i in range(len(Inventory_test)):
    # difference data
    Inventory_months_in_year = 12
    diff = difference(Inventory_history, Inventory_months_in_year)
    #diff2 = diff
    #diff = difference(diff2, Inventory_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit( trend='nc',disp=0,)
    yhat = model_fit.forecast()[0]
    #yhat = inverse_difference(diff2, yhat, Inventory_months_in_year)
    yhat = inverse_difference(Inventory_history, yhat, Inventory_months_in_year) 
    Inventory_predictions.append(yhat)
    # observation
    obs = yhat
    Inventory_history.append(obs)
# report performance
mse = mean_squared_error(Inventory_test, Inventory_predictions)
rmse = sqrt(mse)
print('RMSE: %.4f' % rmse)

# errors
residuals = [Inventory_test[i]-Inventory_predictions[i] for i in range(len(Inventory_test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()

############################################################# Prediction
 
#Getting data
import datetime
from dateutil.relativedelta import relativedelta
Data = Series.from_csv('Og_Inventory.csv', header=0)
Inventory_New_Data=Data
last_data_date = Data.iloc[-1:].index[0]
print ('Last data was collected ', last_data_date)
last_data_date.date()
first_data_date = pd.Timestamp(last_data_date.date()+ relativedelta(months=1))
periodicity = 'm'
dfFuture = pd.DataFrame({ 'day': pd.Series([first_data_date, datetime.datetime(2019,7,31)])})
dfFuture.set_index('day',inplace=True)
dfFuture = dfFuture.asfreq(periodicity)
n=len(dfFuture)
Inventory_New_Data = Inventory_New_Data.append(dfFuture)
Inventory_New_Data.columns =['Inventory']
Inventory_New_Data


#splitting data n= number of nan values
split_point = len(Inventory_New_Data) - n
dataset, validation = Inventory_New_Data[0:split_point], Inventory_New_Data[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# load and prepare datasets
dataset = Series.from_csv('dataset.csv',header=0)
Inventory_X = dataset.values.astype('float32')
Inventory_history = [x for x in Inventory_X]
validation = Series.from_csv('validation.csv',header=0)
Inventory_y = validation.values.astype('float32')

# create a differenced Inventory_series function
def difference(dataset, interval=1):
	diff = list()
	for i in range(1, len(dataset)):
		value = (dataset[i] - dataset[i - interval])
		diff.append(value)
	return diff
 
# invert differenced value funtion
def inverse_difference(Inventory_history, yhat, interval=1):
	return yhat + Inventory_history[-interval]
 
# make first prediction
Inventory_predictions = list()
Inventory_bias = 0
# rolling forecasts
for i in range(0, len(Inventory_y)):
    # difference data
    Inventory_months_in_year = 12
    diff = difference(Inventory_history, Inventory_months_in_year)
    #diff2 = diff
    #diff = difference(diff2, Inventory_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    #yhat = inverse_difference(diff2, yhat, Inventory_months_in_year)
    yhat = Inventory_bias+inverse_difference(Inventory_history, yhat, Inventory_months_in_year)
    Inventory_predictions.append(yhat)
    # observation
    obs = yhat
    Inventory_history.append(obs)

Inventory_predictions
Inventory_New_Data
for i in range(0,n):
    Inventory_New_Data.iloc[-n+i:] = Inventory_predictions[i]
    
Inventory_New_Data.index.name='Date'
Pred_Inventory=Inventory_New_Data
Inventory_New_Data

#################################################################Median Price
#Split into dataset and validation
MedianPrice_Data = Series.from_csv('Og_MedianPrice.csv', header=0)
MedianPrice_Data
MedianPrice_split_point = len(MedianPrice_Data) - 5
MedianPrice_dataset, MedianPrice_validation = MedianPrice_Data[0:MedianPrice_split_point], MedianPrice_Data[MedianPrice_split_point:]
print('Dataset %d, Validation %d' % (len(MedianPrice_dataset), len(MedianPrice_validation)))
MedianPrice_dataset.to_csv('dataset.csv')
MedianPrice_validation.to_csv('validation.csv')

############# SUMMARY STATISTICS

MedianPrice_series = Series.from_csv('dataset.csv')
print(MedianPrice_series.describe())

#line plot
from matplotlib import pyplot
MedianPrice_series.plot()
pyplot.show()

#histogram
pyplot.hist(MedianPrice_series)
pyplot.show()

#seasonal line plots
from pandas import DataFrame
from pandas import TimeGrouper
MedianPrice_groups = MedianPrice_series['2012':'2017'].groupby(TimeGrouper('A'))
MedianPrice_years = DataFrame()
pyplot.figure()
i = 1
n_MedianPrice_groups = len(MedianPrice_groups)
for name, group in MedianPrice_groups:
	pyplot.subplot((n_MedianPrice_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()

#density plot

pyplot.figure(1)
pyplot.subplot(211)
MedianPrice_series.hist()
pyplot.subplot(212)
MedianPrice_series.plot(kind='kde')
pyplot.show()

#box and whisker plot

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
MedianPrice_series = Series.from_csv('dataset.csv')
MedianPrice_groups = MedianPrice_series['2012':'2017'].groupby(TimeGrouper('A'))
MedianPrice_years = DataFrame()
for name, group in MedianPrice_groups:
	MedianPrice_years[name.year] = group.values
MedianPrice_years.boxplot()
pyplot.show()

from statsmodels.tsa.seasonal import seasonal_decompose
MedianPrice_result = seasonal_decompose(MedianPrice_series, model='multiplicative')
print(MedianPrice_result.trend)
print(MedianPrice_result.seasonal)
print(MedianPrice_result.resid)
print(MedianPrice_result.observed)
MedianPrice_result.plot()
pyplot.show()

############## NAIVE FORECAST PERFORMANCE
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
MedianPrice_series = Series.from_csv('dataset.csv')
# prepare data
MedianPrice_X = MedianPrice_series.values
MedianPrice_X = MedianPrice_X.astype('float32')
MedianPrice_train_size = int(len(MedianPrice_X) * 0.50)
MedianPrice_train, MedianPrice_test = MedianPrice_X[0:MedianPrice_train_size], MedianPrice_X[MedianPrice_train_size:]
# walk-forward validation for past 9 values
MedianPrice_history = [x for x in MedianPrice_train]
MedianPrice_predictions = list()
for i in range(len(MedianPrice_test)):
	# predict
	yhat = MedianPrice_history[-6]
	MedianPrice_predictions.append(yhat)
	# observation
	obs = MedianPrice_test[i]
	MedianPrice_history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
mse = mean_squared_error(MedianPrice_test, MedianPrice_predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)


############ARIMA for seasonal data

from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
 
# create a differenced MedianPrice_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(MedianPrice_history, yhat, interval=1):
	return yhat + MedianPrice_history[-interval]

# Double difference data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
MedianPrice_months_in_year = 1
stationary = difference(MedianPrice_X, MedianPrice_months_in_year)
stationary.index = MedianPrice_series.index[MedianPrice_months_in_year:]

MedianPrice_months_in_year = 1
stationary = difference(stationary, MedianPrice_months_in_year)
stationary.index = MedianPrice_series.index[MedianPrice_months_in_year+1:]
stationary

# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
# plot
stationary.plot()
pyplot.show()

#ACF AND PACF PLOTS

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
MedianPrice_series = Series.from_csv('stationary.csv')
pyplot.figure()
pyplot.subplot(211)
plot_acf(MedianPrice_series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(MedianPrice_series, ax=pyplot.gca())
pyplot.show()

#Arima
# create a differenced MedianPrice_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(MedianPrice_history, yhat, interval=1):
	return yhat + MedianPrice_history[-interval]


# load data
MedianPrice_series = Series.from_csv('dataset.csv')
# prepare data
MedianPrice_X = MedianPrice_series.values
MedianPrice_X = MedianPrice_X.astype('float32')
MedianPrice_train_size = int(len(MedianPrice_X) * 0.50)
MedianPrice_train, MedianPrice_test = MedianPrice_X[0:MedianPrice_train_size], MedianPrice_X[MedianPrice_train_size:]

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

# walk-forward validation(re-run from here)
MedianPrice_history = [x for x in MedianPrice_train]
MedianPrice_predictions = list()
for i in range(len(MedianPrice_test)):
    # difference data
    MedianPrice_months_in_year = 1
    diff = difference(MedianPrice_history, MedianPrice_months_in_year)
    diff2 = diff
    diff = difference(diff2, MedianPrice_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit( trend='nc',disp=0,)
    yhat = model_fit.forecast()[0]
    yhat = inverse_difference(diff2, yhat, MedianPrice_months_in_year)
    yhat = inverse_difference(MedianPrice_history, yhat, MedianPrice_months_in_year) 
    MedianPrice_predictions.append(yhat)
    # observation
    obs = yhat
    MedianPrice_history.append(obs)
# report performance
mse = mean_squared_error(MedianPrice_test, MedianPrice_predictions)
rmse = sqrt(mse)
print('RMSE: %.4f' % rmse)

# errors
residuals = [MedianPrice_test[i]-MedianPrice_predictions[i] for i in range(len(MedianPrice_test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()

############################################################# Prediction
 
#Getting data
import datetime
from dateutil.relativedelta import relativedelta
Data = Series.from_csv('Og_MedianPrice.csv', header=0)
MedianPrice_New_Data=Data
last_data_date = Data.iloc[-1:].index[0]
print ('Last data was collected ', last_data_date)
last_data_date.date()
first_data_date = pd.Timestamp(last_data_date.date()+ relativedelta(months=1))
periodicity = 'm'
dfFuture = pd.DataFrame({ 'day': pd.Series([first_data_date, datetime.datetime(2019,7,31)])})
dfFuture.set_index('day',inplace=True)
dfFuture = dfFuture.asfreq(periodicity)
n=len(dfFuture)
MedianPrice_New_Data = MedianPrice_New_Data.append(dfFuture)
MedianPrice_New_Data.columns =['MedianPrice']
MedianPrice_New_Data


#splitting data n= number of nan values
split_point = len(MedianPrice_New_Data) - n
dataset, validation = MedianPrice_New_Data[0:split_point], MedianPrice_New_Data[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# load and prepare datasets
dataset = Series.from_csv('dataset.csv',header=0)
MedianPrice_X = dataset.values.astype('float32')
MedianPrice_history = [x for x in MedianPrice_X]
validation = Series.from_csv('validation.csv',header=0)
MedianPrice_y = validation.values.astype('float32')

# create a differenced MedianPrice_series function
def difference(dataset, interval=1):
	diff = list()
	for i in range(1, len(dataset)):
		value = (dataset[i] - dataset[i - 1])
		diff.append(value)
	return diff
 
# invert differenced value funtion
def inverse_difference(MedianPrice_history, yhat, interval=1):
	return yhat + MedianPrice_history[-interval]
 
# make first prediction
MedianPrice_predictions = list()
MedianPrice_bias = 0
# rolling forecasts
for i in range(0, len(MedianPrice_y)):
    # difference data
    MedianPrice_months_in_year = 12
    diff = difference(MedianPrice_history, MedianPrice_months_in_year)
    #diff2 = diff
    #diff = difference(diff2, MedianPrice_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    #yhat = inverse_difference(diff2, yhat, MedianPrice_months_in_year)
    yhat = MedianPrice_bias+inverse_difference(MedianPrice_history, yhat, MedianPrice_months_in_year)
    MedianPrice_predictions.append(yhat)
    # observation
    obs = yhat
    MedianPrice_history.append(obs)

MedianPrice_predictions
MedianPrice_New_Data
for i in range(0,n):
    MedianPrice_New_Data.iloc[-n+i:] = MedianPrice_predictions[i]
    
MedianPrice_New_Data.index.name='Date'
Pred_MedianPrice=MedianPrice_New_Data
MedianPrice_New_Data

#######################################################################MortgageRate
#Split into dataset and validation
MortgageRate_Data = Series.from_csv('Og_MortgageRate.csv', header=0)
MortgageRate_Data
MortgageRate_split_point = len(MortgageRate_Data) - 6
MortgageRate_dataset, MortgageRate_validation = MortgageRate_Data[0:MortgageRate_split_point], MortgageRate_Data[MortgageRate_split_point:]
print('Dataset %d, Validation %d' % (len(MortgageRate_dataset), len(MortgageRate_validation)))
MortgageRate_dataset.to_csv('dataset.csv')
MortgageRate_validation.to_csv('validation.csv')

############# SUMMARY STATISTICS

MortgageRate_series = Series.from_csv('dataset.csv')
print(MortgageRate_series.describe())

#line plot
from matplotlib import pyplot
MortgageRate_series.plot()
pyplot.show()

#histogram
pyplot.hist(MortgageRate_series)
pyplot.show()

#seasonal line plots
from pandas import DataFrame
from pandas import TimeGrouper
MortgageRate_groups = MortgageRate_series['2012':'2017'].groupby(TimeGrouper('A'))
MortgageRate_years = DataFrame()
pyplot.figure()
i = 1
n_MortgageRate_groups = len(MortgageRate_groups)
for name, group in MortgageRate_groups:
	pyplot.subplot((n_MortgageRate_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()

#density plot

pyplot.figure(1)
pyplot.subplot(211)
MortgageRate_series.hist()
pyplot.subplot(212)
MortgageRate_series.plot(kind='kde')
pyplot.show()

#box and whisker plot

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
MortgageRate_series = Series.from_csv('dataset.csv')
MortgageRate_groups = MortgageRate_series['2012':'2017'].groupby(TimeGrouper('A'))
MortgageRate_years = DataFrame()
for name, group in MortgageRate_groups:
	MortgageRate_years[name.year] = group.values
MortgageRate_years.boxplot()
pyplot.show()

from statsmodels.tsa.seasonal import seasonal_decompose
MortgageRate_result = seasonal_decompose(MortgageRate_series, model='multiplicative')
print(MortgageRate_result.trend)
print(MortgageRate_result.seasonal)
print(MortgageRate_result.resid)
print(MortgageRate_result.observed)
MortgageRate_result.plot()
pyplot.show()

############## NAIVE FORECAST PERFORMANCE
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
MortgageRate_series = Series.from_csv('dataset.csv')
# prepare data
MortgageRate_X = MortgageRate_series.values
MortgageRate_X = MortgageRate_X.astype('float32')
MortgageRate_train_size = int(len(MortgageRate_X) * 0.50)
MortgageRate_train, MortgageRate_test = MortgageRate_X[0:MortgageRate_train_size], MortgageRate_X[MortgageRate_train_size:]
# walk-forward validation for past 9 values
MortgageRate_history = [x for x in MortgageRate_train]
MortgageRate_predictions = list()
for i in range(len(MortgageRate_test)):
	# predict
	yhat = MortgageRate_history[-6]
	MortgageRate_predictions.append(yhat)
	# observation
	obs = MortgageRate_test[i]
	MortgageRate_history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
mse = mean_squared_error(MortgageRate_test, MortgageRate_predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)


############ARIMA for seasonal data

from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
 
# create a differenced MortgageRate_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(MortgageRate_history, yhat, interval=1):
	return yhat + MortgageRate_history[-interval]

# Double difference data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
MortgageRate_months_in_year = 1
stationary = difference(MortgageRate_X, MortgageRate_months_in_year)
stationary.index = MortgageRate_series.index[MortgageRate_months_in_year:]

"""MortgageRate_months_in_year = 1
stationary = difference(stationary, MortgageRate_months_in_year)
stationary.index = MortgageRate_series.index[MortgageRate_months_in_year+1:]
stationary"""

# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
# plot
stationary.plot()
pyplot.show()

#ACF AND PACF PLOTS

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
MortgageRate_series = Series.from_csv('stationary.csv')
pyplot.figure()
pyplot.subplot(211)
plot_acf(MortgageRate_series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(MortgageRate_series, ax=pyplot.gca())
pyplot.show()

#Arima
# create a differenced MortgageRate_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(MortgageRate_history, yhat, interval=1):
	return yhat + MortgageRate_history[-interval]


# load data
MortgageRate_series = Series.from_csv('dataset.csv')
# prepare data
MortgageRate_X = MortgageRate_series.values
MortgageRate_X = MortgageRate_X.astype('float32')
MortgageRate_train_size = int(len(MortgageRate_X) * 0.50)
MortgageRate_train, MortgageRate_test = MortgageRate_X[0:MortgageRate_train_size], MortgageRate_X[MortgageRate_train_size:]

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
# walk-forward validation(re-run from here)
MortgageRate_history = [x for x in MortgageRate_train]
MortgageRate_predictions = list()
for i in range(len(MortgageRate_test)):
    # difference data
    MortgageRate_months_in_year = 1
    diff = difference(MortgageRate_history, MortgageRate_months_in_year)
    #diff2 = diff
    #diff = difference(diff2, MortgageRate_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit( trend='nc',disp=0,)
    yhat = model_fit.forecast()[0]
    #yhat = inverse_difference(diff2, yhat, MortgageRate_months_in_year)
    yhat = inverse_difference(MortgageRate_history, yhat, MortgageRate_months_in_year) 
    MortgageRate_predictions.append(yhat)
    # observation
    obs = yhat
    MortgageRate_history.append(obs)
# report performance
mse = mean_squared_error(MortgageRate_test, MortgageRate_predictions)
rmse = sqrt(mse)
print('RMSE: %.4f' % rmse)

# errors
residuals = [MortgageRate_test[i]-MortgageRate_predictions[i] for i in range(len(MortgageRate_test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()

############################################################# Prediction
 
#Getting data
import datetime
from dateutil.relativedelta import relativedelta
Data = Series.from_csv('Og_MortgageRate.csv', header=0)
MortgageRate_New_Data=Data
last_data_date = Data.iloc[-1:].index[0]
print ('Last data was collected ', last_data_date)
last_data_date.date()
first_data_date = pd.Timestamp(last_data_date.date()+ relativedelta(months=1))
periodicity = 'm'
dfFuture = pd.DataFrame({ 'day': pd.Series([first_data_date, datetime.datetime(2019,7,31)])})
dfFuture.set_index('day',inplace=True)
dfFuture = dfFuture.asfreq(periodicity)
n=len(dfFuture)
MortgageRate_New_Data = MortgageRate_New_Data.append(dfFuture)
MortgageRate_New_Data.columns =['MortgageRate']
MortgageRate_New_Data


#splitting data n= number of nan values
split_point = len(MortgageRate_New_Data) - n
dataset, validation = MortgageRate_New_Data[0:split_point], MortgageRate_New_Data[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# load and prepare datasets
dataset = Series.from_csv('dataset.csv',header=0)
MortgageRate_X = dataset.values.astype('float32')
MortgageRate_history = [x for x in MortgageRate_X]
validation = Series.from_csv('validation.csv',header=0)
MortgageRate_y = validation.values.astype('float32')

# create a differenced MortgageRate_series function
def difference(dataset, interval=1):
	diff = list()
	for i in range(1, len(dataset)):
		value = (dataset[i] - dataset[i - 1])
		diff.append(value)
	return diff
 
# invert differenced value funtion
def inverse_difference(MortgageRate_history, yhat, interval=1):
	return yhat + MortgageRate_history[-interval]
 
# make first prediction
MortgageRate_predictions = list()
MortgageRate_bias = 0
# rolling forecasts
for i in range(0, len(MortgageRate_y)):
    # difference data
    MortgageRate_months_in_year = 1
    diff = difference(MortgageRate_history, MortgageRate_months_in_year)
    #diff2 = diff
    #diff = difference(diff2, MortgageRate_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    #yhat = inverse_difference(diff2, yhat, MortgageRate_months_in_year)
    yhat = MortgageRate_bias+inverse_difference(MortgageRate_history, yhat, MortgageRate_months_in_year)
    MortgageRate_predictions.append(yhat)
    # observation
    obs = yhat
    MortgageRate_history.append(obs)

MortgageRate_predictions
MortgageRate_New_Data
for i in range(0,n):
    MortgageRate_New_Data.iloc[-n+i:] = MortgageRate_predictions[i]
    
MortgageRate_New_Data.index.name='Date'
Pred_MortgageRate=MortgageRate_New_Data
MortgageRate_New_Data

#####################################################################NewListings
#Split into dataset and validation
NewListings_Data = Series.from_csv('Og_NewListings.csv', header=0)
NewListings_Data
NewListings_split_point = len(NewListings_Data) - 5
NewListings_dataset, NewListings_validation = NewListings_Data[0:NewListings_split_point], NewListings_Data[NewListings_split_point:]
print('Dataset %d, Validation %d' % (len(NewListings_dataset), len(NewListings_validation)))
NewListings_dataset.to_csv('dataset.csv')
NewListings_validation.to_csv('validation.csv')

############# SUMMARY STATISTICS

NewListings_series = Series.from_csv('dataset.csv')
print(NewListings_series.describe())

#line plot
from matplotlib import pyplot
NewListings_series.plot()
pyplot.show()

#histogram
pyplot.hist(NewListings_series)
pyplot.show()

#seasonal line plots
from pandas import DataFrame
from pandas import TimeGrouper
NewListings_groups = NewListings_series['2012':'2017'].groupby(TimeGrouper('A'))
NewListings_years = DataFrame()
pyplot.figure()
i = 1
n_NewListings_groups = len(NewListings_groups)
for name, group in NewListings_groups:
	pyplot.subplot((n_NewListings_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()

#density plot

pyplot.figure(1)
pyplot.subplot(211)
NewListings_series.hist()
pyplot.subplot(212)
NewListings_series.plot(kind='kde')
pyplot.show()

#box and whisker plot

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
NewListings_series = Series.from_csv('dataset.csv')
NewListings_groups = NewListings_series['2012':'2017'].groupby(TimeGrouper('A'))
NewListings_years = DataFrame()
for name, group in NewListings_groups:
	NewListings_years[name.year] = group.values
NewListings_years.boxplot()
pyplot.show()

from statsmodels.tsa.seasonal import seasonal_decompose
NewListings_result = seasonal_decompose(NewListings_series, model='multiplicative')
print(NewListings_result.trend)
print(NewListings_result.seasonal)
print(NewListings_result.resid)
print(NewListings_result.observed)
NewListings_result.plot()
pyplot.show()

############## NAIVE FORECAST PERFORMANCE
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
NewListings_series = Series.from_csv('dataset.csv')
# prepare data
NewListings_X = NewListings_series.values
NewListings_X = NewListings_X.astype('float32')
NewListings_train_size = int(len(NewListings_X) * 0.50)
NewListings_train, NewListings_test = NewListings_X[0:NewListings_train_size], NewListings_X[NewListings_train_size:]
# walk-forward validation for past 9 values
NewListings_history = [x for x in NewListings_train]
NewListings_predictions = list()
for i in range(len(NewListings_test)):
	# predict
	yhat = NewListings_history[-6]
	NewListings_predictions.append(yhat)
	# observation
	obs = NewListings_test[i]
	NewListings_history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
mse = mean_squared_error(NewListings_test, NewListings_predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)


############ARIMA for seasonal data

from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
 
# create a differenced NewListings_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(NewListings_history, yhat, interval=1):
	return yhat + NewListings_history[-interval]

# Double difference data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
NewListings_months_in_year = 1
stationary = difference(NewListings_X, NewListings_months_in_year)
stationary.index = NewListings_series.index[NewListings_months_in_year:]

NewListings_months_in_year = 1
stationary = difference(stationary, NewListings_months_in_year)
stationary.index = NewListings_series.index[NewListings_months_in_year+1:]
stationary

# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
# plot
stationary.plot()
pyplot.show()

#ACF AND PACF PLOTS

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
NewListings_series = Series.from_csv('stationary.csv')
pyplot.figure()
pyplot.subplot(211)
plot_acf(NewListings_series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(NewListings_series, ax=pyplot.gca())
pyplot.show()

#Arima
# create a differenced NewListings_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(NewListings_history, yhat, interval=1):
	return yhat + NewListings_history[-interval]


# load data
NewListings_series = Series.from_csv('dataset.csv')
# prepare data
NewListings_X = NewListings_series.values
NewListings_X = NewListings_X.astype('float32')
NewListings_train_size = int(len(NewListings_X) * 0.50)
NewListings_train, NewListings_test = NewListings_X[0:NewListings_train_size], NewListings_X[NewListings_train_size:]

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

# walk-forward validation(re-run from here)
NewListings_history = [x for x in NewListings_train]
NewListings_predictions = list()
for i in range(len(NewListings_test)):
    # difference data
    NewListings_months_in_year = 1
    diff = difference(NewListings_history, NewListings_months_in_year)
    diff2 = diff
    diff = difference(diff2, NewListings_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit( trend='nc',disp=0,)
    yhat = model_fit.forecast()[0]
    yhat = inverse_difference(diff2, yhat, NewListings_months_in_year)
    yhat = inverse_difference(NewListings_history, yhat, NewListings_months_in_year) 
    NewListings_predictions.append(yhat)
    # observation
    obs = yhat
    NewListings_history.append(obs)
    # report performance
mse = mean_squared_error(NewListings_test, NewListings_predictions)
rmse = sqrt(mse)
print('RMSE: %.4f' % rmse)

# errors
residuals = [NewListings_test[i]-NewListings_predictions[i] for i in range(len(NewListings_test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()

############################################################# Prediction
 
#Getting data
import datetime
from dateutil.relativedelta import relativedelta
Data = Series.from_csv('Og_NewListings.csv', header=0)
NewListings_New_Data=Data
last_data_date = Data.iloc[-1:].index[0]
print ('Last data was collected ', last_data_date)
last_data_date.date()
first_data_date = pd.Timestamp(last_data_date.date()+ relativedelta(months=1))
periodicity = 'm'
dfFuture = pd.DataFrame({ 'day': pd.Series([first_data_date, datetime.datetime(2019,7,31)])})
dfFuture.set_index('day',inplace=True)
dfFuture = dfFuture.asfreq(periodicity)
n=len(dfFuture)
NewListings_New_Data = NewListings_New_Data.append(dfFuture)
NewListings_New_Data.columns =['NewListings']
NewListings_New_Data


#splitting data n= number of nan values
split_point = len(NewListings_New_Data) - n
dataset, validation = NewListings_New_Data[0:split_point], NewListings_New_Data[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# load and prepare datasets
dataset = Series.from_csv('dataset.csv',header=0)
NewListings_X = dataset.values.astype('float32')
NewListings_history = [x for x in NewListings_X]
validation = Series.from_csv('validation.csv',header=0)
NewListings_y = validation.values.astype('float32')

# create a differenced NewListings_series function
def difference(dataset, interval=1):
	diff = list()
	for i in range(1, len(dataset)):
		value = (dataset[i] - dataset[i - 1])
		diff.append(value)
	return diff
 
# invert differenced value funtion
def inverse_difference(NewListings_history, yhat, interval=1):
	return yhat + NewListings_history[-interval]
 
# make first prediction
NewListings_predictions = list()
NewListings_bias = 0
# rolling forecasts
for i in range(0, len(NewListings_y)):
    # difference data
    NewListings_months_in_year = 12
    diff = difference(NewListings_history, NewListings_months_in_year)
    #diff2 = diff
    #diff = difference(diff2, NewListings_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    #yhat = inverse_difference(diff2, yhat, NewListings_months_in_year)
    yhat = NewListings_bias+inverse_difference(NewListings_history, yhat, NewListings_months_in_year)
    NewListings_predictions.append(yhat)
    # observation
    obs = yhat
    NewListings_history.append(obs)

NewListings_predictions
NewListings_New_Data
for i in range(0,n):
    NewListings_New_Data.iloc[-n+i:] = NewListings_predictions[i]
    
NewListings_New_Data.index.name='Date'
Pred_NewListings=NewListings_New_Data
NewListings_New_Data

#######################################################################Population
#Split into dataset and validation
Population_Data = Series.from_csv('Og_Population.csv', header=0)
Population_Data
Population_split_point = len(Population_Data) - 5
Population_dataset, Population_validation = Population_Data[0:Population_split_point], Population_Data[Population_split_point:]
print('Dataset %d, Validation %d' % (len(Population_dataset), len(Population_validation)))
Population_dataset.to_csv('dataset.csv')
Population_validation.to_csv('validation.csv')

############# SUMMARY STATISTICS

Population_series = Series.from_csv('dataset.csv')
print(Population_series.describe())

#line plot
from matplotlib import pyplot
Population_series.plot()
pyplot.show()

#histogram
pyplot.hist(Population_series)
pyplot.show()

#seasonal line plots
from pandas import DataFrame
from pandas import TimeGrouper
Population_groups = Population_series['2012':'2017'].groupby(TimeGrouper('A'))
Population_years = DataFrame()
pyplot.figure()
i = 1
n_Population_groups = len(Population_groups)
for name, group in Population_groups:
	pyplot.subplot((n_Population_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()

#density plot

pyplot.figure(1)
pyplot.subplot(211)
Population_series.hist()
pyplot.subplot(212)
Population_series.plot(kind='kde')
pyplot.show()

#box and whisker plot
"""
from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
Population_series = Series.from_csv('dataset.csv')
Population_groups = Population_series['2012':'2017'].groupby(TimeGrouper('A'))
Population_years = DataFrame()
for name, group in Population_groups:
	Population_years[name.year] = group.values
Population_years.boxplot()
pyplot.show()"""

from statsmodels.tsa.seasonal import seasonal_decompose
Population_result = seasonal_decompose(Population_series, model='multiplicative')
print(Population_result.trend)
print(Population_result.seasonal)
print(Population_result.resid)
print(Population_result.observed)
Population_result.plot()
pyplot.show()

############## NAIVE FORECAST PERFORMANCE
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
Population_series = Series.from_csv('dataset.csv')
# prepare data
Population_X = Population_series.values
Population_X = Population_X.astype('float32')
Population_train_size = int(len(Population_X) * 0.50)
Population_train, Population_test = Population_X[0:Population_train_size], Population_X[Population_train_size:]
# walk-forward validation for past 9 values
Population_history = [x for x in Population_train]
Population_predictions = list()
for i in range(len(Population_test)):
	# predict
	yhat = Population_history[-6]
	Population_predictions.append(yhat)
	# observation
	obs = Population_test[i]
	Population_history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
mse = mean_squared_error(Population_test, Population_predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)


############ARIMA for seasonal data

from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
 
# create a differenced Population_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(Population_history, yhat, interval=1):
	return yhat + Population_history[-interval]

# Double difference data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
Population_months_in_year = 1
stationary = difference(Population_X, Population_months_in_year)
stationary.index = Population_series.index[Population_months_in_year:]

Population_months_in_year = 1
stationary = difference(stationary, Population_months_in_year)
stationary.index = Population_series.index[Population_months_in_year+1:]
stationary

# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
# plot
stationary.plot()
pyplot.show()

#ACF AND PACF PLOTS

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
Population_series = Series.from_csv('stationary.csv')
pyplot.figure()
pyplot.subplot(211)
plot_acf(Population_series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(Population_series, ax=pyplot.gca())
pyplot.show()

#Arima
# create a differenced Population_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(Population_history, yhat, interval=1):
	return yhat + Population_history[-interval]


# load data
Population_series = Series.from_csv('dataset.csv')
# prepare data
Population_X = Population_series.values
Population_X = Population_X.astype('float32')
Population_train_size = int(len(Population_X) * 0.50)
Population_train, Population_test = Population_X[0:Population_train_size], Population_X[Population_train_size:]

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

# walk-forward validation(re-run from here)
Population_history = [x for x in Population_train]
Population_predictions = list()
for i in range(len(Population_test)):
    # difference data
    Population_months_in_year = 1
    diff = difference(Population_history, Population_months_in_year)
    diff2 = diff
    diff = difference(diff2, Population_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit( trend='nc',disp=0,)
    yhat = model_fit.forecast()[0]
    yhat = inverse_difference(diff2, yhat, Population_months_in_year)
    yhat = inverse_difference(Population_history, yhat, Population_months_in_year) 
    Population_predictions.append(yhat)
    # observation
    obs = yhat
    Population_history.append(obs)
# report performance
mse = mean_squared_error(Population_test, Population_predictions)
rmse = sqrt(mse)
print('RMSE: %.4f' % rmse)

# errors
residuals = [Population_test[i]-Population_predictions[i] for i in range(len(Population_test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()

############################################################# Prediction
 
#Getting data
import datetime
from dateutil.relativedelta import relativedelta
Data = Series.from_csv('Og_Population.csv', header=0)
Population_New_Data=Data
last_data_date = Data.iloc[-1:].index[0]
print ('Last data was collected ', last_data_date)
last_data_date.date()
first_data_date = pd.Timestamp(last_data_date.date()+ relativedelta(months=1))
periodicity = 'm'
dfFuture = pd.DataFrame({ 'day': pd.Series([first_data_date, datetime.datetime(2019,7,31)])})
dfFuture.set_index('day',inplace=True)
dfFuture = dfFuture.asfreq(periodicity)
n=len(dfFuture)
Population_New_Data = Population_New_Data.append(dfFuture)
Population_New_Data.columns =['Population']
Population_New_Data


#splitting data n= number of nan values
split_point = len(Population_New_Data) - n
dataset, validation = Population_New_Data[0:split_point], Population_New_Data[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# load and prepare datasets
dataset = Series.from_csv('dataset.csv',header=0)
Population_X = dataset.values.astype('float32')
Population_history = [x for x in Population_X]
validation = Series.from_csv('validation.csv',header=0)
Population_y = validation.values.astype('float32')

# create a differenced Population_series function
def difference(dataset, interval=1):
	diff = list()
	for i in range(1, len(dataset)):
		value = (dataset[i] - dataset[i - 1])
		diff.append(value)
	return diff
 
# invert differenced value funtion
def inverse_difference(Population_history, yhat, interval=1):
	return yhat + Population_history[-interval]
 
# make first prediction
Population_predictions = list()
Population_bias = 0
# rolling forecasts
for i in range(0, len(Population_y)):
    # difference data
    Population_months_in_year = 1
    diff = difference(Population_history, Population_months_in_year)
    diff2 = diff
    diff = difference(diff2, Population_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    yhat = inverse_difference(diff2, yhat, Population_months_in_year)
    yhat = Population_bias+inverse_difference(Population_history, yhat, Population_months_in_year)
    Population_predictions.append(yhat)
    # observation
    obs = yhat
    Population_history.append(obs)

Population_predictions
Population_New_Data
for i in range(0,n):
    Population_New_Data.iloc[-n+i:] = Population_predictions[i]
    
Population_New_Data.index.name='Date'
Pred_Population=Population_New_Data
Population_New_Data

###############################################################################################################33#PrimeRate
#Split into dataset and validation
PrimeRate_Data = Series.from_csv('Og_PrimeRate.csv', header=0)
PrimeRate_Data
PrimeRate_split_point = len(PrimeRate_Data) - 6
PrimeRate_dataset, PrimeRate_validation = PrimeRate_Data[0:PrimeRate_split_point], PrimeRate_Data[PrimeRate_split_point:]
print('Dataset %d, Validation %d' % (len(PrimeRate_dataset), len(PrimeRate_validation)))
PrimeRate_dataset.to_csv('dataset.csv')
PrimeRate_validation.to_csv('validation.csv')

############# SUMMARY STATISTICS

PrimeRate_series = Series.from_csv('dataset.csv')
print(PrimeRate_series.describe())

#line plot
from matplotlib import pyplot
PrimeRate_series.plot()
pyplot.show()

#histogram
pyplot.hist(PrimeRate_series)
pyplot.show()

#seasonal line plots
from pandas import DataFrame
from pandas import TimeGrouper
PrimeRate_groups = PrimeRate_series['2012':'2017'].groupby(TimeGrouper('A'))
PrimeRate_years = DataFrame()
pyplot.figure()
i = 1
n_PrimeRate_groups = len(PrimeRate_groups)
for name, group in PrimeRate_groups:
	pyplot.subplot((n_PrimeRate_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()

#density plot

pyplot.figure(1)
pyplot.subplot(211)
PrimeRate_series.hist()
pyplot.subplot(212)
PrimeRate_series.plot(kind='kde')
pyplot.show()

#box and whisker plot

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
PrimeRate_series = Series.from_csv('dataset.csv')
PrimeRate_groups = PrimeRate_series['2012':'2017'].groupby(TimeGrouper('A'))
PrimeRate_years = DataFrame()
for name, group in PrimeRate_groups:
	PrimeRate_years[name.year] = group.values
PrimeRate_years.boxplot()
pyplot.show()

from statsmodels.tsa.seasonal import seasonal_decompose
PrimeRate_result = seasonal_decompose(PrimeRate_series, model='multiplicative')
print(PrimeRate_result.trend)
print(PrimeRate_result.seasonal)
print(PrimeRate_result.resid)
print(PrimeRate_result.observed)
PrimeRate_result.plot()
pyplot.show()

############## NAIVE FORECAST PERFORMANCE
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
PrimeRate_series = Series.from_csv('dataset.csv')
# prepare data
PrimeRate_X = PrimeRate_series.values
PrimeRate_X = PrimeRate_X.astype('float32')
PrimeRate_train_size = int(len(PrimeRate_X) * 0.50)
PrimeRate_train, PrimeRate_test = PrimeRate_X[0:PrimeRate_train_size], PrimeRate_X[PrimeRate_train_size:]
# walk-forward validation for past 9 values
PrimeRate_history = [x for x in PrimeRate_train]
PrimeRate_predictions = list()
for i in range(len(PrimeRate_test)):
	# predict
	yhat = PrimeRate_history[-6]
	PrimeRate_predictions.append(yhat)
	# observation
	obs = PrimeRate_test[i]
	PrimeRate_history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
mse = mean_squared_error(PrimeRate_test, PrimeRate_predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)


############ARIMA for seasonal data

from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
 
# create a differenced PrimeRate_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(PrimeRate_history, yhat, interval=1):
	return yhat + PrimeRate_history[-interval]

# Double difference data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
PrimeRate_months_in_year = 1
stationary = difference(PrimeRate_X, PrimeRate_months_in_year)
stationary.index = PrimeRate_series.index[PrimeRate_months_in_year:]

PrimeRate_months_in_year = 1
stationary = difference(stationary, PrimeRate_months_in_year)
stationary.index = PrimeRate_series.index[PrimeRate_months_in_year+1:]
stationary

# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
# plot
stationary.plot()
pyplot.show()

#ACF AND PACF PLOTS

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
PrimeRate_series = Series.from_csv('stationary.csv')
pyplot.figure()
pyplot.subplot(211)
plot_acf(PrimeRate_series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(PrimeRate_series, ax=pyplot.gca())
pyplot.show()

#Arima
# create a differenced PrimeRate_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(PrimeRate_history, yhat, interval=1):
	return yhat + PrimeRate_history[-interval]


# load data
PrimeRate_series = Series.from_csv('dataset.csv')
# prepare data
PrimeRate_X = PrimeRate_series.values
PrimeRate_X = PrimeRate_X.astype('float32')
PrimeRate_train_size = int(len(PrimeRate_X) * 0.50)
PrimeRate_train, PrimeRate_test = PrimeRate_X[0:PrimeRate_train_size], PrimeRate_X[PrimeRate_train_size:]

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
# walk-forward validation(re-run from here)
PrimeRate_history = [x for x in PrimeRate_train]
PrimeRate_predictions = list()
for i in range(len(PrimeRate_test)):
    # difference data
    PrimeRate_months_in_year = 1
    diff = difference(PrimeRate_history, PrimeRate_months_in_year)
    diff2 = diff
    diff = difference(diff2, PrimeRate_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit( trend='nc',disp=0,)
    yhat = model_fit.forecast()[0]
    yhat = inverse_difference(diff2, yhat, PrimeRate_months_in_year)
    yhat = inverse_difference(PrimeRate_history, yhat, PrimeRate_months_in_year) 
    PrimeRate_predictions.append(yhat)
    # observation
    obs = yhat
    PrimeRate_history.append(obs)
# report performance
mse = mean_squared_error(PrimeRate_test, PrimeRate_predictions)
rmse = sqrt(mse)
print('RMSE: %.4f' % rmse)

# errors
residuals = [PrimeRate_test[i]-PrimeRate_predictions[i] for i in range(len(PrimeRate_test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()

############################################################# Prediction
 
#Getting data
import datetime
from dateutil.relativedelta import relativedelta
Data = Series.from_csv('Og_PrimeRate.csv', header=0)
PrimeRate_New_Data=Data
last_data_date = Data.iloc[-1:].index[0]
print ('Last data was collected ', last_data_date)
last_data_date.date()
first_data_date = pd.Timestamp(last_data_date.date()+ relativedelta(months=1))
periodicity = 'm'
dfFuture = pd.DataFrame({ 'day': pd.Series([first_data_date, datetime.datetime(2019,7,31)])})
dfFuture.set_index('day',inplace=True)
dfFuture = dfFuture.asfreq(periodicity)
n=len(dfFuture)
PrimeRate_New_Data = PrimeRate_New_Data.append(dfFuture)
PrimeRate_New_Data.columns =['PrimeRate']
PrimeRate_New_Data


#splitting data n= number of nan values
split_point = len(PrimeRate_New_Data) - n
dataset, validation = PrimeRate_New_Data[0:split_point], PrimeRate_New_Data[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# load and prepare datasets
dataset = Series.from_csv('dataset.csv',header=0)
PrimeRate_X = dataset.values.astype('float32')
PrimeRate_history = [x for x in PrimeRate_X]
validation = Series.from_csv('validation.csv',header=0)
PrimeRate_y = validation.values.astype('float32')

# create a differenced PrimeRate_series function
def difference(dataset, interval=1):
	diff = list()
	for i in range(1, len(dataset)):
		value = (dataset[i] - dataset[i - 1])
		diff.append(value)
	return diff
 
# invert differenced value funtion
def inverse_difference(PrimeRate_history, yhat, interval=1):
	return yhat + PrimeRate_history[-interval]
 
# make first prediction
PrimeRate_predictions = list()
PrimeRate_bias = 0
# rolling forecasts
for i in range(0, len(PrimeRate_y)):
    # difference data
    PrimeRate_months_in_year = 1
    diff = difference(PrimeRate_history, PrimeRate_months_in_year)
    #diff2 = diff
    #diff = difference(diff2, PrimeRate_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    #yhat = inverse_difference(diff2, yhat, PrimeRate_months_in_year)
    yhat = PrimeRate_bias+inverse_difference(PrimeRate_history, yhat, PrimeRate_months_in_year)
    PrimeRate_predictions.append(yhat)
    # observation
    obs = yhat
    PrimeRate_history.append(obs)

PrimeRate_predictions
PrimeRate_New_Data
for i in range(0,n):
    PrimeRate_New_Data.iloc[-n+i:] = PrimeRate_predictions[i]
    
PrimeRate_New_Data.index.name='Date'
Pred_PrimeRate=PrimeRate_New_Data
PrimeRate_New_Data

#################################################################SalesVolume
#Split into dataset and validation
SalesVolume_Data = Series.from_csv('Og_SalesVolume.csv', header=0)
SalesVolume_Data
SalesVolume_split_point = len(SalesVolume_Data) - 5
SalesVolume_dataset, SalesVolume_validation = SalesVolume_Data[0:SalesVolume_split_point], SalesVolume_Data[SalesVolume_split_point:]
print('Dataset %d, Validation %d' % (len(SalesVolume_dataset), len(SalesVolume_validation)))
SalesVolume_dataset.to_csv('dataset.csv')
SalesVolume_validation.to_csv('validation.csv')

############# SUMMARY STATISTICS

SalesVolume_series = Series.from_csv('dataset.csv')
print(SalesVolume_series.describe())

#line plot
from matplotlib import pyplot
SalesVolume_series.plot()
pyplot.show()

#histogram
pyplot.hist(SalesVolume_series)
pyplot.show()

#seasonal line plots
from pandas import DataFrame
from pandas import TimeGrouper
SalesVolume_groups = SalesVolume_series['2012':'2017'].groupby(TimeGrouper('A'))
SalesVolume_years = DataFrame()
pyplot.figure()
i = 1
n_SalesVolume_groups = len(SalesVolume_groups)
for name, group in SalesVolume_groups:
	pyplot.subplot((n_SalesVolume_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()

#density plot

pyplot.figure(1)
pyplot.subplot(211)
SalesVolume_series.hist()
pyplot.subplot(212)
SalesVolume_series.plot(kind='kde')
pyplot.show()

#box and whisker plot

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
SalesVolume_series = Series.from_csv('dataset.csv')
SalesVolume_groups = SalesVolume_series['2012':'2017'].groupby(TimeGrouper('A'))
SalesVolume_years = DataFrame()
for name, group in SalesVolume_groups:
	SalesVolume_years[name.year] = group.values
SalesVolume_years.boxplot()
pyplot.show()

from statsmodels.tsa.seasonal import seasonal_decompose
SalesVolume_result = seasonal_decompose(SalesVolume_series, model='multiplicative')
print(SalesVolume_result.trend)
print(SalesVolume_result.seasonal)
print(SalesVolume_result.resid)
print(SalesVolume_result.observed)
SalesVolume_result.plot()
pyplot.show()

############## NAIVE FORECAST PERFORMANCE
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
SalesVolume_series = Series.from_csv('dataset.csv')
# prepare data
SalesVolume_X = SalesVolume_series.values
SalesVolume_X = SalesVolume_X.astype('float32')
SalesVolume_train_size = int(len(SalesVolume_X) * 0.50)
SalesVolume_train, SalesVolume_test = SalesVolume_X[0:SalesVolume_train_size], SalesVolume_X[SalesVolume_train_size:]
# walk-forward validation for past 9 values
SalesVolume_history = [x for x in SalesVolume_train]
SalesVolume_predictions = list()
for i in range(len(SalesVolume_test)):
	# predict
	yhat = SalesVolume_history[-6]
	SalesVolume_predictions.append(yhat)
	# observation
	obs = SalesVolume_test[i]
	SalesVolume_history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
mse = mean_squared_error(SalesVolume_test, SalesVolume_predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)


############ARIMA for seasonal data

from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot
 
# create a differenced SalesVolume_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(SalesVolume_history, yhat, interval=1):
	return yhat + SalesVolume_history[-interval]

# Double difference data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
SalesVolume_months_in_year = 1
stationary = difference(SalesVolume_X, SalesVolume_months_in_year)
stationary.index = SalesVolume_series.index[SalesVolume_months_in_year:]

SalesVolume_months_in_year = 1
stationary = difference(stationary, SalesVolume_months_in_year)
stationary.index = SalesVolume_series.index[SalesVolume_months_in_year+1:]
stationary

# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
# plot
stationary.plot()
pyplot.show()

#ACF AND PACF PLOTS

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
SalesVolume_series = Series.from_csv('stationary.csv')
pyplot.figure()
pyplot.subplot(211)
plot_acf(SalesVolume_series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(SalesVolume_series, ax=pyplot.gca())
pyplot.show()

#Arima
# create a differenced SalesVolume_series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced value
def inverse_difference(SalesVolume_history, yhat, interval=1):
	return yhat + SalesVolume_history[-interval]


# load data
SalesVolume_series = Series.from_csv('dataset.csv')
# prepare data
SalesVolume_X = SalesVolume_series.values
SalesVolume_X = SalesVolume_X.astype('float32')
SalesVolume_train_size = int(len(SalesVolume_X) * 0.50)
SalesVolume_train, SalesVolume_test = SalesVolume_X[0:SalesVolume_train_size], SalesVolume_X[SalesVolume_train_size:]

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

# walk-forward validation(re-run from here)
SalesVolume_history = [x for x in SalesVolume_train]
SalesVolume_predictions = list()
for i in range(len(SalesVolume_test)):
    # difference data
    SalesVolume_months_in_year = 1
    diff = difference(SalesVolume_history, SalesVolume_months_in_year)
    diff2 = diff
    diff = difference(diff2, SalesVolume_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit( trend='nc',disp=0,)
    yhat = model_fit.forecast()[0]
    yhat = inverse_difference(diff2, yhat, SalesVolume_months_in_year)
    yhat = inverse_difference(SalesVolume_history, yhat, SalesVolume_months_in_year) 
    SalesVolume_predictions.append(yhat)
    # observation
    obs = yhat
    SalesVolume_history.append(obs)
# report performance
mse = mean_squared_error(SalesVolume_test, SalesVolume_predictions)
rmse = sqrt(mse)
print('RMSE: %.4f' % rmse)

# errors
residuals = [SalesVolume_test[i]-SalesVolume_predictions[i] for i in range(len(SalesVolume_test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()

############################################################# Prediction
 
#Getting data
import datetime
from dateutil.relativedelta import relativedelta
Data = Series.from_csv('Og_SalesVolume.csv', header=0)
SalesVolume_New_Data=Data
last_data_date = Data.iloc[-1:].index[0]
print ('Last data was collected ', last_data_date)
last_data_date.date()
first_data_date = pd.Timestamp(last_data_date.date()+ relativedelta(months=1))
periodicity = 'm'
dfFuture = pd.DataFrame({ 'day': pd.Series([first_data_date, datetime.datetime(2019,7,31)])})
dfFuture.set_index('day',inplace=True)
dfFuture = dfFuture.asfreq(periodicity)
n=len(dfFuture)
SalesVolume_New_Data = SalesVolume_New_Data.append(dfFuture)
SalesVolume_New_Data.columns =['SalesVolume']
SalesVolume_New_Data


#splitting data n= number of nan values
split_point = len(SalesVolume_New_Data) - n
dataset, validation = SalesVolume_New_Data[0:split_point], SalesVolume_New_Data[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')

# load and prepare datasets
dataset = Series.from_csv('dataset.csv',header=0)
SalesVolume_X = dataset.values.astype('float32')
SalesVolume_history = [x for x in SalesVolume_X]
validation = Series.from_csv('validation.csv',header=0)
SalesVolume_y = validation.values.astype('float32')

# create a differenced SalesVolume_series function
def difference(dataset, interval=1):
	diff = list()
	for i in range(1, len(dataset)):
		value = (dataset[i] - dataset[i - 1])
		diff.append(value)
	return diff
 
# invert differenced value funtion
def inverse_difference(SalesVolume_history, yhat, interval=1):
	return yhat + SalesVolume_history[-interval]
 
# make first prediction
SalesVolume_predictions = list()
SalesVolume_bias = 0
# rolling forecasts
for i in range(0, len(SalesVolume_y)):
    # difference data
    SalesVolume_months_in_year = 12
    diff = difference(SalesVolume_history, SalesVolume_months_in_year)
    #diff2 = diff
    #diff = difference(diff2, SalesVolume_months_in_year)
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    #yhat = inverse_difference(diff2, yhat, SalesVolume_months_in_year)
    yhat = SalesVolume_bias+inverse_difference(SalesVolume_history, yhat, SalesVolume_months_in_year)
    SalesVolume_predictions.append(yhat)
    # observation
    obs = yhat
    SalesVolume_history.append(obs)

SalesVolume_predictions
SalesVolume_New_Data
for i in range(0,n):
    SalesVolume_New_Data.iloc[-n+i:] = SalesVolume_predictions[i]
    
SalesVolume_New_Data.index.name='Date'
Pred_SalesVolume=SalesVolume_New_Data
SalesVolume_New_Data

####################################################################################################Combine All

df1=pd.merge(Pred_Inflation,Pred_Income,how='inner', left_index=True, right_index=True)
df2=pd.merge(df1,Pred_CPI,how='inner', left_index=True, right_index=True)
df3=pd.merge(df2,Pred_Inventory,how='inner', left_index=True, right_index=True)
df4=pd.merge(df3,Pred_SalesVolume,how='inner', left_index=True, right_index=True)
df5=pd.merge(df4,Pred_MortgageRate,how='inner', left_index=True, right_index=True)
df6=pd.merge(df5,Pred_NewListings,how='inner', left_index=True, right_index=True)
df7=pd.merge(df6,Pred_Population,how='inner', left_index=True, right_index=True)
df8=pd.merge(df7,Pred_PrimeRate,how='inner', left_index=True, right_index=True)
df=pd.merge(df8,Pred_MedianPrice,how='inner', left_index=True, right_index=True)

#To Excel
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('All_Pred_Columba.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
