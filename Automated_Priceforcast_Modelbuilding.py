# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 17:38:20 2018

@author: Adrien
"""


from pymongo import MongoClient
import pymongo
import datetime
import pprint
import numpy as np
import pandas as pd
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime
from dateutil.relativedelta import relativedelta


# create a differenced series function
def difference(dataset, interval):
	diff = list()
	for i in range(1, len(dataset)):
		value = (dataset[i] - dataset[i - interval])
		diff.append(value)
	return diff

# invert differenced value function
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def Mongo_Inv(Inv_Data,firstdate,lastdate):
    Mongo_Inv = pd.DataFrame({ 'day': pd.Series([firstdate, lastdate])})
    Mongo_Inv.set_index('day',inplace=True)
    Mongo_Inv = Mongo_Inv.asfreq('m')
    Inv_Data = Inv_Data.set_index(Mongo_Inv.index)
    return Inv_Data

def TS_Inv(Inv_Data):
    last_data_date = Inv_Data.iloc[-1:].index[0]
    print ('Last data was collected ', last_data_date)
    last_data_date.date()
    first_data_date = pd.Timestamp(last_data_date.date()+ relativedelta(months=1))
    dfFuture = pd.DataFrame({ 'day': pd.Series([first_data_date, datetime.datetime(2019,8,31)])})
    dfFuture.set_index('day',inplace=True)
    dfFuture = dfFuture.asfreq('m')
    n=len(dfFuture)
    Inv_Data = Inv_Data.append(dfFuture)
    Inv_Data.columns =['Inventory']
    Inv_Data
    
    #splitting data
    split_point = len(Inv_Data) - n
    dataset, validation = Inv_Data[0:split_point], Inv_Data[split_point:]
    print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
    dataset.to_csv('dataset.csv')
    validation.to_csv('validation.csv')
    
    # load and prepare datasets
    dataset = Series.from_csv('dataset.csv',header=0)
    X = dataset.values.astype('float32')
    history = [x for x in X]
    validation = Series.from_csv('validation.csv',header=0)
    y = validation.values.astype('float32')
    
    # make first prediction
    predictions = list()
    bias = 0
        
    #walk-forward validation-Using ARIMA(1,0,0) 
    
    for i in range(len(y)):
        # difference data once by 12months
        months_in_year = 12
        diff = difference(history, months_in_year)
        #second differencing by one month
        #diff2=diff
        #months_in_year = 1
        #diff=difference(diff2,months_in_year)
        # predict
        try:
            model=ARIMA(diff, order=(2,0,0))
            model_fit = model.fit( trend='nc',disp=0,)
        except ValueError:
            model=ARIMA(diff, order=(1,0,0))
            model_fit = model.fit( trend='nc',disp=0,)
            
            
        yhat = model_fit.forecast()[0]
        #inverting once
        #yhat = inverse_difference(diff2, yhat, months_in_year)
        #months_in_year = 12
        #second inverse
        yhat = bias+inverse_difference(history, yhat, months_in_year)
        predictions.append(yhat)
        # updating observation for next step
        obs = yhat
        history.append(obs)
    
    predictions
    Inv_Data
    for i in range(0,n):
        Inv_Data.iloc[-n+i:] = predictions[i]
    
    Inv_Data.index.name='Date'
    return Inv_Data
         
    
def Mongo_Price(Price_data,firstdate,lastdate):
    Mongo_Price = pd.DataFrame({ 'day': pd.Series([firstdate, lastdate])})
    Mongo_Price.set_index('day',inplace=True)
    Mongo_Price = Mongo_Price.asfreq('m')
    Price_data=Price_data.set_index(Mongo_Price.index)
    return Price_data
    
def Inv_All(All,Inv_Data):
    All['Inventory_DB']=Inv_Data['Inventory'].values
    return All

def TS_Price(Price_data):
    last_data_date = Price_data.iloc[-1:].index[0]
    print ('Last data was collected ', last_data_date)
    last_data_date.date()
    first_data_date = pd.Timestamp(last_data_date.date()+ relativedelta(months=1))
    dfFuture = pd.DataFrame({ 'date': pd.Series([first_data_date, datetime.datetime(2019,8,31)])})
    dfFuture.set_index('date',inplace=True)
    dfFuture = dfFuture.asfreq('m')
    n=len(dfFuture)
    Price_data = Price_data.append(dfFuture)
    Price_data.columns =['Price_DB']
    Price_data
    
    #Splitting median price data
    split_point = len(Price_data) - n
    dataset, validation = Price_data[0:split_point], Price_data[split_point:]
    print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
    dataset.to_csv('dataset.csv')
    validation.to_csv('validation.csv')
    
    # load and prepare median price datasets
    dataset = Series.from_csv('dataset.csv',header=0)
    X = dataset.values.astype('float32')
    history = [x for x in X]
    months_in_year = 1
    validation = Series.from_csv('validation.csv',header=0)
    y = validation.values.astype('float32')
    
    
    ##### make prediction for median price till august 2019 
    #Adding Bias to get the mean error to 0
    bias=2563
    predictions = list()
    for i in range(len(y)):
        # difference data once by one month
        months_in_year = 12
        diff = difference(history, months_in_year)
        # predict
        model = ARIMA(diff, order=(1,0,0))
        model_fit = model.fit( trend='nc',disp=0,)
        yhat = model_fit.forecast()[0]
        #Inversing data once
        yhat = bias+inverse_difference(history, yhat, months_in_year) 
        predictions.append(yhat)
        # updating the observation
        obs = yhat
        history.append(obs)
        
    predictions
    #adding predictions of median price from august 2018 to august 2019 to Price_data
    Price_data
    for i in range(0,n):
        Price_data.iloc[-n+i:] = predictions[i]
    
    Price_data.index.name='Date'
    return Price_data

def Price_All(All,Price_data):
    All['Price_DB']=Price_data['Price_DB'].values
    return All
    

def Mongo_data_extractor(dom_data,Inv_Data,Price_data,key,zipcode,county,inventory_value,median_listprice_value):
    dom_data = db.dom_data.find({'zipcode':zipcode}) 
    for f in dom_data:
        Approach3_data = f["Approach3"]                                        
        for i in range(inventory_value,-1,-1):
            firstdate=Approach3_data[key][0]['Inventory'][inventory_value]['Datetime']
            lastdate=Approach3_data[key][0]['Inventory'][i]['Datetime']
            l=int(Approach3_data[key][0]['Inventory'][i]['Inventory'])
            Inv_Data.iloc[(inventory_value-i),:]=l
            
            
            
            
    ###########################Extract MongoDB Inventory data
    Inv_Data=Mongo_Inv(Inv_Data,firstdate,lastdate)
    
    
    ###########################TimeSeries of InventoryData
    Inv_Data=TS_Inv(Inv_Data)
     
    ####################################Adding Inventory to Pred_Data
    All= pd.read_excel('All_Pred_%s.xlsx'%county,parse_dates=['Date'],index_col='Date')
    All=All.iloc[-(inventory_value+1+19):,:]
    All=Inv_All(All,Inv_Data)    
    
    
    ############################################################################Median price from MongoDB
    dom_data = db.dom_data.find({'zipcode':zipcode})    
    for f in dom_data:
        Approach3_data = f["Approach3"]
        print ("Approach3_data")
        print (Approach3_data)
        for i in range(median_listprice_value,-1,-1):
            print ("jdhfgdjhfg")
            print (median_listprice_value)
            firstdate=Approach3_data[key][0]['Option1'][median_listprice_value]['date_start']
            lastdate=Approach3_data[key][0]['Option1'][i]['date_start']
            print ("ndhdhfh")
            print (key)
            print (Approach3_data[key][0]['Option1'])
            print (Approach3_data[key][0]['Option1'][i]['median_listprice'])
            Price=int(Approach3_data[key][0]['Option1'][i]['median_listprice'])
            Price_data.iloc[median_listprice_value-i,:]=Price   
    
    ###############################Extracting Medianprice from mongoDB
    Price_data= Mongo_Price(Price_data,firstdate,lastdate)


    
    #################################Timeseries of Medianprice
    Price_data=TS_Price(Price_data)
    
    ###############Adding Price to Pred_Data dataframe
    All_value=Price_All(All,Price_data)
    All.index = All_value.index.astype(str)
    Dataframe_value = All.to_dict('index')
    return All_value,Dataframe_value

##############################################################################################SVR Function

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def pickle(A):
    import pickle
    A = pickle.dumps(A)
    return A

def model(All):
    #Get data
    print ("value")
    X = All.iloc[:,0:11].values
    print ("am x")
    print (X)
    y = All.iloc[:,[11]].values
    print ("am y")
    print (y)
    #Feature Scaling
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    sc_y = StandardScaler()
    y = sc_y.fit_transform(y)
    #SVR: Predicting with input values
    svr = SVR(kernel='linear', C=20, epsilon=0.001)
    A=svr.fit(X,y)
    A = pickle(A)
    sc_X = pickle(sc_X)
    sc_y = pickle(sc_y)
    return A,sc_X,sc_y




def value_validation(value):
        inventry = []
        median_price = []
        g = value[0]["Inventory"]
        price = value[0]["Option1"]
        for index in range(len(g)):
            #print (g[index])
            k =g[index]
            #print (k)
            #print (index)
            if index == 61:
                k.update ({"Supply":g[index]["Inventory"]})
            else:
                k.update ({"Supply":((g[index]["Inventory"] +g[index+1]["Inventory"])/2)})
            inventry.append(k)
        for index in range(len(price)):
            #print (g[index])
            #print (price[index])
            median_price.append(price[index])


            
            
        #print (inventry)
        #print (median_price)
        dataframe_inventory = pd.DataFrame(inventry)
        dataframe_inventory = dataframe_inventory[:20]
        dataframe_inventory = dataframe_inventory[(dataframe_inventory.Supply > 1)]
        inventory_value = len(dataframe_inventory.index)
        #print(inventory_value ) 
        dataframe_median_price = pd.DataFrame(median_price)
        dataframe_median_price_new = dataframe_median_price[:20]
        print ("gehghgergrgr")
        print (dataframe_median_price_new)
        dataframe_inventory = dataframe_median_price_new[(dataframe_median_price_new.median_listprice > 1)]
        median_listprice_value = len(dataframe_inventory.index)
        #print (median_listprice_value)
        return(inventory_value,median_listprice_value)



def modelfilewriter(dict_value,db,collection_name):
    #db[collection_name].insert(dict_value)
    db['%s'%collection_name].find_one_and_update({"zipcode":dict_value["zipcode"]},
                                                                    update={"$set": dict_value},
                                                                    upsert=True, full_response= True)
    #db['%s'%collection_name].find_and_modify({"zipcode":dict_value["zipcode"]},
                                                                    #update={"$set": dict_value},
                                                                    #upsert=True, full_response= True)
    return True




################ Extract Inventory from MongoDB
uri = "mongodb:**********Your database url****************************"
client = MongoClient(uri)
db = client.***YOUR_DB_NAME******
dom_data = db.***DATA***.find({"zipcode" : 97236}) 

for val in dom_data:
      
    approach = val["Approach3"]

    
    
    zipcode = val["zipcode"]
    county = val["county"]
    #for key,val in approach.iteritems:
            #print (key)
    result_dict_outer = {}
    result_data_refrennce1 = {}
    result_dict = {}
    for key, value in approach.items():
        print (zipcode)
        print (key)
        Inv_Data=pd.DataFrame(index=[list(range(21))],columns=["COLUMN_Inv"])   
        Price_data=pd.DataFrame(index=[list(range(21))],columns=["COLUMN_Price"])
        
        
        if key =='Message':
            result_dict.update({key:{"message":"some data point missing for model building"}})
        else :    
            inventory_value,median_listprice_value = value_validation(value)
            if inventory_value == 21 and median_listprice_value == 21:
                Inv_Data=pd.DataFrame(index=[list(range(inventory_value+1))],columns=["COLUMN_Inv"])   
                Price_data=pd.DataFrame(index=[list(range(median_listprice_value+1))],columns=["COLUMN_Price"])
                All,Dataframe_value = Mongo_data_extractor(dom_data,Inv_Data,Price_data,key,zipcode,county,inventory_value,median_listprice_value)
                
                
                ### Returns model and scaled input
                #model1,sc_X,sc_y = model(All)
                
                
                X = All.iloc[:,0:11].values
                print ("am x")
                print (X)
                y = All.iloc[:,[11]].values
                print ("am y")
                print (y)
                #Feature Scaling
                sc_X = StandardScaler()
                X = sc_X.fit_transform(X)
                sc_y = StandardScaler()
                y = sc_y.fit_transform(y)
                #SVR: Predicting with input values
                svr = SVR(kernel='linear', C=20, epsilon=0.001)
                A=svr.fit(X,y)
                model1 = pickle(A)
                sc_X = pickle(sc_X)
                sc_y = pickle(sc_y)
    
        
                result_dict.update({key:{"model":model1,"sc_X":sc_X,"sc_y":sc_y,"extrapolated":Dataframe_value}})
            else:
                
                result_dict.update({key:{"message":"some data point missing for model building"}})


    resultdict = {"county":county,"zipcode":zipcode,"model":result_dict}

    collection_name = 'forcast_model'
    #collection_name_resultdict = "model_reference"
    modelfilewriter(resultdict,db,collection_name)

