# -*- coding: utf-8 -*-
"""
Created on June 15 - dev ver2.0
@author: MR004CHM
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import pydot
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn import linear_model

tf.random.set_seed(777)
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

def test_day_when(DAY):
    table=pd.read_csv('workdays.csv')
    testday=DAY-260
    return(table.iloc[testday,1])

def scaler_set(data_ref):
    global scalerX
    scalerX = MinMaxScaler(feature_range=(0.001,1))     ## to avoid ZERO inputs
    scalerX.fit(data_ref[:,0:(n_input)])                ## to cover the WHOLE SEASONS

    
def train_NN(data_ready, n_epoch):
    global records_train, nn_hist
    data_train = data_ready[train_start_day*daymin:test_start_day*daymin,0:(n_input+1)]
    X_var = data_train[:,0:(n_input)]
    Y_var = data_train[:,(n_input)]/1000000
    trainX, valX, trainY, valY = train_test_split(X_var, Y_var, test_size=0.1, shuffle=True, random_state=777)
    trainX = scalerX.transform(trainX)
    valX   = scalerX.transform(valX)

    model = keras.Sequential()
    model.add(layers.Dense(32, input_dim=(n_input), activation="sigmoid", name="layer1"))
    model.add(layers.Dense(32,activation="sigmoid", name="layer2"))
    model.add(layers.Dense(1, name="output"))
    model.compile(loss='MSE', optimizer='adam')
    model.summary()
    nn_hist = model.fit(trainX, trainY, epochs=n_epoch, batch_size=5000)
#    trainScore = np.sqrt(model.evaluate(trainX, trainY, verbose=0))
    model.reset_states()
#    valScore   = np.sqrt(model.evaluate(valX, valY, verbose=0))
#    model.reset_states()
#    SCORES = pd.DataFrame({'train_score' : trainScore, 'validation_score': valScore, 'train_date' : }, index=[0])
#    records_train = records_train.append(SCORES, ignore_index=True)
    return(model)

def validation_visual_NN(data_ready, n_epoch, case_name, TrainDay, TestDay):
    data_train = data_ready[TrainDay*daymin:TestDay*daymin,0:(n_input+1)]
    X_var = data_train[:,0:(n_input)]
    Y_var = data_train[:,(n_input)]/1000000
    trainX, valX, trainY, valY = train_test_split(X_var, Y_var, test_size=0.1, shuffle=True, random_state=777)
    trainX = scalerX.transform(trainX)
    valX   = scalerX.transform(valX)

    model = keras.Sequential()
    model.add(layers.Dense(32, input_dim=(n_input), activation="sigmoid", name="layer1"))
    model.add(layers.Dense(32,activation="sigmoid", name="layer2"))
    model.add(layers.Dense(1, name="output"))
    model.compile(loss='MSE', optimizer='adam')
    model.summary()
    
    nn_hist = model.fit(trainX, trainY, epochs=n_epoch, batch_size=10000)
    trainScore = np.sqrt(model.evaluate(trainX, trainY, verbose=0))
    model.reset_states()
    valScore   = np.sqrt(model.evaluate(valX, valY, verbose=0))
    model.reset_states()

    #plot_model(model, to_file='./model.png', show_shapes=(True), show_layer_names=(False))
    plt.plot(np.sqrt(nn_hist.history['loss'])) #RMSE PLOT
    #plt.ylim(0.0, 0.5)
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    print(("train error: {}. validation error: {}".format(trainScore, valScore)))


def predict_NN(NN_MODEL,data_ready):
    global records_test, RMSE_matrix
    model = NN_MODEL
    data_test  = data_ready[test_start_day*daymin:(test_start_day+5*test_len)*daymin , 0:(n_input+1)]
    testX = data_test[:,0:n_input]
    testY = data_test[:,n_input]/1000000
    testX = scalerX.transform(testX)
    y_pred  = model.predict(testX)
    y_pred[y_pred < 0] = 0                # non-negative for RMSLE estimation
    y_true  = testY.reshape(-1,1)
    result_error = np.sqrt(mean_squared_log_error(y_true,y_pred))
    result_error = round(result_error,3)
    SCORES = pd.DataFrame({'test_score' : result_error, 'train_date': train_start_day , 'test_date' : test_start_day}, index=[0])
    records_test = records_test.append(SCORES, ignore_index=True)
    RMSE_matrix.iloc[i//5,j//5] = result_error


def predict_visual_NN(NN_MODEL,data_ready):
    global records_test
    model = NN_MODEL
    data_test  = data_ready[test_start_day*daymin:(test_start_day+5*test_len)*daymin , 0:(n_input+1)]
    testX = data_test[:,0:n_input]
    testY = data_test[:,n_input]/1000000
    testX = scalerX.transform(testX)

    y_pred  = model.predict(testX)
    y_true  = testY.reshape(-1,1)
    result_rmse = sklearn.metrics.mean_squared_error(y_true,y_pred,squared=False)
    result_rmse = round(result_rmse,3)
    #RMSE_matrix.iloc[i,j] = result_rmse
    SCORES = pd.DataFrame({'test_score' : result_rmse, 'train_date': train_start_day , 'test_date' : test_start_day}, index=[0])
    records_test = records_test.append(SCORES, ignore_index=True)
    
    plt.figure(dpi=500) 
    plt.plot(y_true, color='blue')
    plt.plot(y_pred, color='black')
    plt.ylabel('Energy(MJ)')
    plt.xlabel('min')
    plt.legend(['Measured','Predicted'], loc='upper left')
    plt.title("Training: 52weeks from {}th day \n Prediction: {}weeks from {}(th)day \n RMSE= {}".format(train_start_day, test_len, test_start_day,result_rmse ))
    plt.show()
    

def records_save():
    now = datetime.datetime.now()
    run_date = now.strftime('%m%d_%H%M')
    trainsheet = "records_train_" + run_date +".csv"
    testsheet  = "records_test_" + run_date +".csv"
    RMSEsheet  = "RMSE_matrix_" + run_date +".csv"
    records_train.to_csv(trainsheet, index=False)
    records_test.to_csv(testsheet, index=False)
    RMSE_matrix.to_csv(RMSEsheet, index=False)
    


######################################################################################################################
#%% Variable Setting ####

## data filter variables
startup   = 30             # filter for start-up period to remove the simulation noise
daymin    = 60*11-startup
n_input   = 4

# operation varialbes
test_len= 1
n_iter  = 260//(test_len*5)

# Record Initialization
records_train = pd.DataFrame({'train_score':[0], 'validation_score':[0], 'train_date':[0]})
records_test  = pd.DataFrame({'test_score':[0],'train_date':[0], 'test_date':[0]})
RMSE_matrix = pd.DataFrame(np.zeros((n_iter, n_iter)))

######################################################################################################################
#%% DATA Preparation

DATA_NAME = 'chiller_unfaulted_HK.csv'
# Data Import
data_from_csv = pd.read_csv(DATA_NAME)
data_from_csv.set_axis(['OA_temp','inlet_flowrate','inlet_temp','outlet_temp','electric_energy'],axis='columns', inplace=True)
# Data Filter : weekdays & operation hour(07-18)
data_ref = filter_optime(data_from_csv, 'weekend.csv')
data_ref = filter_startup(data_ref, startup)
data_ref = data_ref.to_numpy()
#optimesheet  = ("filtered_" + DATA_NAME)
#data_optime.to_csv(optimesheet, index=False)

DATA_NAME = 'chiller_fouling_MAC.csv'
# Data Import
data_from_csv = pd.read_csv(DATA_NAME)
data_from_csv.set_axis(['OA_temp','inlet_flowrate','inlet_temp','outlet_temp','electric_energy'],axis='columns', inplace=True)
# Data Filter : weekdays & operation hour(07-18)
data_foul = filter_optime(data_from_csv, 'weekend.csv')
data_foul = filter_startup(data_foul, startup)
data_foul = data_foul.to_numpy()
#optimesheet  = ("filtered_" + DATA_NAME)
#data_optime.to_csv(optimesheet, index=False)


data_tot   = np.vstack([data_ref, data_foul])
scaler_set(data_tot)


#%% Sliding-Window RUN

for i in range(0,260,5):
    train_start_day = i
    test_start_day = train_start_day+260   # number of workdays in the calendar is 260
    nn_model_HKMAC = train_NN(data_tot, 250)
    #validation_visual_NN(data_tot, 250, 'unfaulted-HK', train_start_day, test_start_day)
    #nn_model.save('model_normal_HK.h5')
    for j in range(i,260,5):
        test_start_day = 260 + j
        predict_NN(nn_model_HKMAC,data_tot)
        #predict_visual_NN(nn_model_HKMAC,data_tot)
        

#%% Result Analysis
records_save()
heatdata = pd.read_csv('RMSE_matrix_0624_1100.csv')
sns.heatmap(heatdata)



######################################################################################################################
#%% REFERENCE - fixed training model

for i in [0]:
    train_start_day = i
    test_start_day = train_start_day+260   # number of workdays in the calendar is 260
    nn_model_HKMAC = train_NN(data_tot, 250)
    #validation_visual_NN(data_tot, 250, 'unfaulted-HK', train_start_day, test_start_day)
    #nn_model.save('model_normal_HK.h5')
    for j in range(0,260,5):
        test_start_day = train_start_day + 260 + j
        predict_NN(nn_model_HKMAC,data_tot)
        #predict_visual_NN(nn_model_HKMAC,data_tot)
    



data_test  = data_tot[test_start_day*daymin:(test_start_day+5*test_len)*daymin , 0:(n_input+1)]
testX = data_test[:,0:n_input]
testY = data_test[:,n_input]/1000000
testX = scalerX.transform(testX)

y_pred  = nn_model_HKMAC.predict(testX)
y_true  = testY.reshape(-1,1)


