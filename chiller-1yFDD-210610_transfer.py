# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:55:53 2021

@author: MR004CHM
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import pydot
import matplotlib.pyplot as plt
import datetime

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


os.chdir('C:\\Users\\MR004CHM\\Desktop\\TFcode\\2021-fdd')

######################################################################################################################
#%% DEFINE FUNCTIONS ####

def filter_optime(rawdata, week_info_csv):
    # weekdays and weekends (2017y)
    week_info   = pd.read_csv(week_info_csv)
    weekday     = week_info[['week']].to_numpy(dtype='int32')
    weekdaymins = np.repeat(weekday,1440)
    #operation hours: 07-18
    workhour     = np.concatenate((np.zeros(60*7,dtype='int32'),np.ones(60*11,dtype='int32'),np.zeros(60*6,dtype='int32')),axis=None)
    workhourmins = np.tile(workhour, 365)
    #filter : Zeros during deactivated
    optime = weekdaymins*workhourmins
    # APPLY filter
    # CHECK: in 2017, weekdays are 260 out of 365
    data = rawdata
    data['operation']=optime
    data = data[data['operation'] != 0] 
    return(data)

def filter_startup(rawdata, cutoff):
    #cut data at start-up period (IT IS NOISY DATA)
    ndays         = len(rawdata)//(60*11)
    startuptime   = np.concatenate((np.zeros(cutoff,dtype='int32'),np.ones(60*11-cutoff,dtype='int32')),axis=None)
    startupfilter = np.tile(startuptime, ndays)
    # APPLY filter
    # CHECK: in 2017, weekdays are 260 out of 365
    data = rawdata
    data['startups']= startupfilter
    data = data[data['startups'] != 0] 
    return(data)


def start_month(month):
    #month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    work_days = [0, 22, 20, 23, 20, 23, 22, 21, 23, 21, 22, 22, 21]
    past = sum(work_days[0:month])
    total = past
    minute = total*daymin
    return(minute)

def test_day_when(testmin):
    testday=testmin//daymin
    table=pd.read_csv('workdays.csv')
    return(table.iloc[testday,1])


def data_slice(data_ready):
    data = data_ready
    global trainX, trainY, valX, valY, scalerX
    scalerX = MinMaxScaler(feature_range=(0.001,1)) ## to avoid ZERO inputs
    scalerX.fit(data[:,0:(n_input)])                ## to cover the WHOLE SEASONS
    X_var = data[train_startmin:train_endmin,0:(n_input)]
    Y_var = data[train_startmin:train_endmin:,(n_input)]/1000000
    trainX, valX, trainY, valY = train_test_split(X_var, Y_var, test_size=0.1, shuffle=True, random_state=777)
    trainX=scalerX.transform(trainX)
    valX=scalerX.transform(valX)
    
    #### SLICE function AND TRAIN function WILL BE MERGED AFTER ALL THINGS DONE WELL 
    
def train_NN(trainX,trainY,n_epoch):
    global records_train, nn_hist
    model = keras.Sequential()
    model.add(layers.Dense(32, input_dim=(n_input), activation="sigmoid", name="layer1"))
    model.add(layers.Dense(32,activation="sigmoid", name="layer2"))
    model.add(layers.Dense(1, name="output"))
    model.compile(loss='MSE', optimizer='adam')
    model.summary()
    nn_hist = model.fit(trainX, trainY, epochs=n_epoch)
    trainScore = np.sqrt(model.evaluate(trainX, trainY, verbose=0))
    model.reset_states()
    valScore   = np.sqrt(model.evaluate(valX, valY, verbose=0))
    model.reset_states()
    SCORES = pd.DataFrame({'train_score' : trainScore, 'validation_score': valScore, 'train_date' : train_startmin/daymin}, index=[0])
    records_train = records_train.append(SCORES, ignore_index=True)
    return(model)

def FDD_stepwise(NN_MODEL,data_ready):
    global records_test
    data = data_ready
    model=NN_MODEL
    testX = data[test_startmin:test_endmin,0:(n_input)]
    testY = data[test_startmin:test_endmin:,(n_input)]/1000000
    testX=scalerX.transform(testX)
    result_RMSE = np.sqrt(model.evaluate(testX, testY, verbose=0))
    SCORES = pd.DataFrame({'test_score' : result_RMSE, 'train_date': train_startmin/daymin, 'test_date' : test_startmin/daymin}, index=[0])
    records_test = records_test.append(SCORES, ignore_index=True)
    RMSE_matrix.iloc[i,j] = result_RMSE
    
def records_save():
    now = datetime.datetime.now()
    run_date = now.strftime('%m%d_%H%M')
    trainsheet = "records_train_" + run_date +".csv"
    testsheet  = "records_test_" + run_date +".csv"
    RMSEsheet  = "RMSE_matrix_" + run_date +".csv"
    records_train.to_csv(trainsheet, index=False)
    records_test.to_csv(testsheet, index=False)
    RMSE_matrix.to_csv(RMSEsheet, index=False)
    
def visual_NN(mymodel,data_ready):
    data = data_ready
    model = mymodel
    case_name= DATA_NAME[0:-4]
    #plot_model(model, to_file='./model.png', show_shapes=(True), show_layer_names=(False))
    plt.plot(np.sqrt(nn_hist.history['loss'])) #RMSE PLOT
    plt.ylim(0.0, 0.5)
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    testX = data[test_startmin:test_endmin,0:(n_input)]
    testY = data[test_startmin:test_endmin:,(n_input)]/1000000
    testX=scalerX.transform(testX)
    result_predict  = model.predict(testX)
    result_measured = testY.reshape(-1,1)
    test_day = test_day_when(test_startmin)
    train_day = test_day_when(train_startmin)
    plt.figure(dpi=500) 
    plt.plot(result_measured, color='blue')
    plt.plot(result_predict, color='black')
    plt.ylabel('Energy(MJ)')
    plt.xlabel('min')
    plt.legend(['Measured','Expected'], loc='upper left')
    plt.title("Case:{} \n train:4weeks from {}, Result:{}".format(case_name,train_day, test_day))
    plt.show()
    
def visual_weeks(mymodel,data_ready,weekstart,n_weeks):
    data = data_ready
    model = mymodel
    case_name= DATA_NAME[0:-4]
    test_startmin = weekstart  ## Customized test period
    test_endmin   = test_startmin +(5*n_weeks*daymin)     ## N weeks Prediction
    testX = data[test_startmin:test_endmin,0:(n_input)]
    testY = data[test_startmin:test_endmin:,(n_input)]/1000000
    testX=scalerX.transform(testX)
    y_pred  = model.predict(testX)
    y_true = testY.reshape(-1,1)
    result_rmse = sklearn.metrics.mean_squared_error(y_true,y_pred,squared=False)
    result_rmse = round(result_rmse,3)
    test_day = test_day_when(test_startmin)
    train_day = test_day_when(train_startmin)
    plt.figure(dpi=500) 
    plt.plot(y_true, color='blue')
    plt.plot(y_pred, color='black')
    plt.ylabel('Energy(MJ)')
    plt.xlabel('min')
    plt.legend(['Measured','Predicted'], loc='upper left')
    plt.title("Case:{} (RMSE={}) \n Train:4weeks from {}, Predict:{}weeks from {}".format(
        case_name,result_rmse, train_day, n_weeks, test_day))
    plt.show()
    

######################################################################################################################
#%% Variable Setting ####

DATA_NAME = 'chiller_1y_fouling.csv'

# Data Import
data_from_csv = pd.read_csv(DATA_NAME)
data_from_csv.set_axis(['OA_temp','inlet_flowrate','inlet_temp','outlet_temp','electric_energy'],axis='columns', inplace=True)
n_input=4

## SET (very important) variables for analysis
iter_days = 1                  
startup   = 30             # filter for start-up period to remove the simulation noise
daymin    = 60*11-startup
calendar_min  = start_month(1) # type the start MONTH

#Initialization
records_train = pd.DataFrame({'train_score':[0], 'validation_score':[0], 'train_date':[0]})
records_transfer = pd.DataFrame({'train_score':[0], 'validation_score':[0], 'train_date':[0]})
records_test  = pd.DataFrame({'test_score':[0],'train_date':[0], 'test_date':[0]})
RMSE_matrix = pd.DataFrame(np.zeros((iter_days,iter_days)))

# Data Filter : weekdays & operation hour(07-18)
data_optime = filter_optime(data_from_csv, 'weekend.csv')
data_optime = filter_startup(data_optime, startup)
data_optime = data_optime.to_numpy()
#optimesheet  = ("filtered_" + DATA_NAME)
#data_optime.to_csv(optimesheet, index=False)

#%% RUN RUN RUN ####

train_startmin = calendar_min
train_endmin   = calendar_min + (260)*daymin
data_slice(data_optime)
nn_model = train_NN(trainX,trainY,250)


visual_weeks(nn_model, data_optime, 81900,2)  # Apr3:40590, May1: 53550, June5: 69300, July3:81900, 88200, Aug:97650




#nn_model.save('model_chiller.h5')


######################################################################################################################
#%% DEBUG and Development PLANT ####

def transfer_FDD(NN_MODEL, n_epoch, data_ready):
    global tfer_hist 
    addX = data_ready[:, 0:(n_input)]
    addY = data_ready[:, (n_input)]/1000000
    addX = scalerX.transform(addX)
    model = NN_MODEL
    for layer in model.layers[:-1]:
        layer.trainable = False
    model.compile(loss='MSE', optimizer='adam')   
    tfer_hist = model.fit(addX, addY, epochs=n_epoch)
#    trainScore = np.sqrt(model.evaluate(addX, addY, verbose=0))
    model.reset_states()
    return(model)

tfer_data = pd.read_csv('chiller_unfaulted_HK170612.csv')
tfer_data = tfer_data.to_numpy()
tfer_model = nn_model
transfer_FDD(tfer_model, 10, tfer_data)
#tfer_model = transfer_FDD(nn_model, 100, optime_data, 97650)

visual_weeks(tfer_model, data_optime, 81900,2)  # Apr3:40590, May1: 53550, June5: 69300, July3:81900, 88200, Aug:97650





















def FDD_stepwise(NN_MODEL,data_ready):
    global records_test
    model=NN_MODEL
    data  = data_ready.to_numpy()
    testX = data[test_startmin:test_endmin,0:(n_input)]
    testY = data[test_startmin:test_endmin:,(n_input)]/1000000
    testX=scalerX.transform(testX)
    result_RMSE = np.sqrt(model.evaluate(testX, testY, verbose=0))
    SCORES = pd.DataFrame({'test_score' : result_RMSE, 'train_date': train_startmin/daymin, 'test_date' : test_startmin/daymin}, index=[0])
    records_test = records_test.append(SCORES, ignore_index=True)
    RMSE_matrix.iloc[i,j] = result_RMSE

iter_days = 1
calendar_min  = start_month(4) # type the start MONTH





for i in range(0, iter_days):
    print(i)
    train_startmin = calendar_min + i*daymin
    train_endmin   = calendar_min + (i+20)*daymin
    data_slice(data_optime)
    nn_model = train_NN(trainX,trainY,200)
    for j in range (i, iter_days):
        print(j)
        test_startmin = calendar_min + (j+20+5)*daymin
        test_endmin   = calendar_min + (j+20+6)*daymin
        FDD_stepwise(nn_model,data_optime)








# Loop execution
# RULE: 20 days for training, prediction test after 5days
for i in range(0, iter_days):
    print(i)
    train_startmin = calendar_min + i*daymin
    train_endmin   = calendar_min + (i+20)*daymin
    data_slice(data_optime)
    nn_model = train_NN(trainX,trainY,250)
    for j in range (i, iter_days):
        print(j)
        test_startmin = calendar_min + (j+20+5)*daymin
        test_endmin   = calendar_min + (j+20+6)*daymin
        FDD_stepwise(nn_model,data_optime)

# Export the results to CSV in path
# records_save()

# Visualization of TRAIN and TEST loss (dafult: last iteration)
# visual_NN (nn_model, data_optime)
visual_weeks(nn_model, data_optime,81900,2)  # Apr3:40590, May1: 53550, June5: 69300, July3:81900, 88200, Aug:97650





