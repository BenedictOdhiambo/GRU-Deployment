#!/usr/bin/env python
# coding: utf-8

# ## Name: Benedict Odhiambo
# ## Reg no:21/08516
# ## Course:MDA(Data Analytics and Knowledge Engineering)
# ## Task: Assignment 3 and Assignment 4

# ## 1) About Dataset

# Dataset was on Airpassengers from 1949 to 1960, acquired from Kaggle:https://www.kaggle.com/code/ktakuma/air-passengers-prediction-by-rnn-lstm-gru/data

# ## 2) Data Exploration, Preprocessing and choosing Dependent/Independent Variables

# In[1]:


# installing python library for fast numerical computing
get_ipython().system('pip install tensorflow')


# In[2]:


#importing the python libraries to be used
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow import keras # Python library for developing and evaluating deep learning models
import pandas as pd # python library for data science/data analysis and machine learning tasks
from sklearn.metrics import mean_absolute_error,accuracy_score,mean_squared_error,confusion_matrix #Used to measure the perfomance of a model
from keras.models import Sequential # used for analysis and comparison of simple neural network-oriented models
from keras.layers import Dense, LSTM, Dropout, GRU #developing and evaluating deep learning models
import numpy as np
import matplotlib.pyplot as plt #Python library for plotting
import warnings, math
warnings.filterwarnings('ignore')
from keras.callbacks import EarlyStopping


# In[3]:


# Load the dataset
Data=pd.read_csv('AirPassengers.csv', usecols=[1])
Data.tail()


# In[4]:


#Renaming the column names since the hashtag would render it a comment
Data.rename(columns={'#Passengers':'Passengers'}, inplace=True)
Dataset = Data.values
Dataset = Data.astype('float32') #Convert values to float


# Independent Variable-Month
# 
# Dependent Variable-Passengers
# 
# Month affects the number of passengers who book the plane for air travel.
# The independent variable is the variable whose values are manipulated by the investigator to establish its effect on the values of dependent variable 
# Dependent variables are expected to change as a result of an experimental manipulation of the independent variable

# ## 3i) LSTM

# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1)) #QuantileTransformer can also be used
Dataset = scaler.fit_transform(Dataset)
# split into train and test sets
train_size = int(len(Dataset) * 0.7)
test_size = len(Dataset) - train_size
train, test = Dataset[0:train_size,:], Dataset[train_size:len(Dataset),:]


# In[6]:


#creates a dataset where X is the number of passengers at a given time (t, t-1, t-2...) 
#and Y is the number of passengers at the next time (t + 1).

def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        #print(i)
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)
    

seq_size = 20  # Number of time steps to look back 
#Larger sequences (look further back) may improve forecasting.

trainX, trainY = to_sequences(train, seq_size)
testX, testY = to_sequences(test, seq_size)



print("Shape of training set: {}".format(trainX.shape))
print("Shape of test set: {}".format(testX.shape))


# In[7]:


#Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#Apply the lstm algorithm to the dataset and define all the constants and requirements to form the model
print('Single LSTM with hidden Dense...')
model = Sequential()
model.add(LSTM(80, input_shape=(None, seq_size)))
model.add(Dense(40))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, 
                       verbose=2, mode='auto', restore_best_weights=True)
model.summary()
print('Train...')


# In[8]:


#Application of the created model
model.fit(trainX, trainY, validation_data=(testX, testY),
          verbose=1, epochs=2000)


# In[9]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
plt.show()


# #### 5)Test Validation and Perfomance of LSTM

# In[10]:


# invert predictions back to prescaled values
#This is to compare with original input values
#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.
trainPredict1 = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict1 = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error to validate the results
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict1[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict1[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[11]:


# shift train predictions for plotting
#we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot = np.empty_like(Dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[seq_size:len(trainPredict1)+seq_size, :] = trainPredict1
# shift test predictions for plotting
testPredictPlot = np.empty_like(Dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict1)+(seq_size*2)+1:len(Dataset)-1, :] = testPredict1


# In[12]:


# plot baseline and predictions
plt.figure(figsize=(10,8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(scaler.inverse_transform(Dataset),color="r",label="True-result")
plt.plot(trainPredictPlot,color="b",label="Train-predicted-result")
plt.plot(testPredictPlot,color="g",label="Test-Predicted-result")
plt.legend()
plt.title("LSTM PLOT")
plt.xlabel("Time Step(Months)")
plt.ylabel("Number of Air Passengers")
plt.grid(True)
plt.show()


# ## 3ii) GRU

# In[13]:


def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        #print(i)
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)
    

seq_size = 20  # Number of time steps to look back 
#Larger sequences (look further back) may improve forecasting.

train_X, train_Y = to_sequences(train, seq_size)
test_X, test_Y = to_sequences(test, seq_size)



print("Shape of training set: {}".format(train_X.shape))
print("Shape of test set: {}".format(test_X.shape))


# In[14]:


#creates a dataset where X is the number of passengers at a given time (t, t-1, t-2...) 
#and Y is the number of passengers at the next time (t + 1).
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

print('GRU')
gru = Sequential()
gru.add(GRU(80, input_shape=(None, seq_size)))
gru.add(Dense(40))
gru.add(Dense(1))
gru.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, 
                       verbose=2, mode='auto', restore_best_weights=True)
gru.summary()
print('Train...')


# In[15]:


train_Y=np.transpose(train_Y)


# In[16]:


test_Y=np.transpose(test_Y)


# In[17]:


train_X = np.reshape(train_X, (79, -1, 20))
test_X = np.reshape(test_X, (23, -1, 20))


# In[18]:


gru.fit(train_X, train_Y, validation_data=(test_X, test_Y),
        verbose=1, epochs=2000)


# In[19]:


# make predictions
trainPredict_g= gru.predict(train_X)
testPredict_g = gru.predict(test_X)
plt.show()


# In[20]:


trainPredict_g= np.reshape(trainPredict_g, (-1, 1))
testPredict_g = np.reshape(testPredict_g, (-1, 1))


# #### 5)Test, Validation and Perfomance of GRU

# In[21]:


# invert predictions back to prescaled values
#This is to compare with original input values
#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.
trainPredict_g1 = scaler.inverse_transform(trainPredict_g)
train_Y = scaler.inverse_transform([train_Y])
testPredict_g1 = scaler.inverse_transform(testPredict_g)
test_Y = scaler.inverse_transform([test_Y])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_Y[0], trainPredict_g1[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(test_Y[0], testPredict_g1[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[22]:




# shift train predictions for plotting
#we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot_g = np.empty_like(Dataset)
trainPredictPlot_g[:, :] = np.nan
trainPredictPlot_g[seq_size:len(trainPredict_g)+seq_size, :] = trainPredict_g1
# shift test predictions for plotting
testPredictPlot_g = np.empty_like(Dataset)
testPredictPlot_g[:, :] = np.nan
testPredictPlot_g[len(trainPredict_g)+(seq_size*2)+1:len(Dataset)-1, :] = testPredict_g1

# plot baseline and predictions
plt.figure(figsize=(10,8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(scaler.inverse_transform(Dataset),color="r",label="True-result")
plt.plot(trainPredictPlot_g,color="b",label="Train-predicted-result")
plt.plot(testPredictPlot_g,color="g",label="Test-Predicted-result")
plt.legend()
plt.title("GRU PLOT")
plt.xlabel("Time Step(Months)")
plt.ylabel("Number of Air Passengers")
plt.grid(True)
plt.show()


# ## 4) Ensemble Learning with averaging technique

# In[23]:


def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset)-seq_size-1):
        #print(i)
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
        
    return np.array(x),np.array(y)
    

seq_size = 20  # Number of time steps to look back 
#Larger sequences (look further back) may improve forecasting.

train_Xe, train_Ye = to_sequences(train, seq_size)
test_Xe, test_Ye = to_sequences(test, seq_size)



print("Shape of training set: {}".format(train_X.shape))
print("Shape of test set: {}".format(test_X.shape))


# In[24]:


train_Xe = np.reshape(train_Xe, (train_Xe.shape[0], 1, train_Xe.shape[1]))
test_Xe = np.reshape(test_Xe, (test_Xe.shape[0], 1, test_Xe.shape[1]))


# In[25]:


pred_final=(testPredict_g +testPredict)/2.0


# In[26]:


pred_final_train=(trainPredict_g+trainPredict)/2.0


# In[27]:


pred_final_train= np.reshape(pred_final_train, (-1, 1))
pred_final = np.reshape(pred_final, (-1, 1))


# In[28]:


train_Ye=np.transpose(train_Ye)


# In[29]:


test_Ye=np.transpose(test_Ye)


# #### 5)Test,Perfomance and Validation of Ensemble Learning

# In[30]:


# invert predictions back to prescaled values
#This is to compare with original input values
#SInce we used minmaxscaler we can now use scaler.inverse_transform
#to invert the transformation.
pred_final_train= scaler.inverse_transform(pred_final_train)
train_Ye = scaler.inverse_transform([train_Ye])
pred_final = scaler.inverse_transform(pred_final)
test_Ye= scaler.inverse_transform([test_Ye])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_Ye[0], pred_final_train[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(test_Ye[0], pred_final[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[31]:


# shift train predictions for plotting
#we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot_g = np.empty_like(Dataset)
trainPredictPlot_g[:, :] = np.nan
trainPredictPlot_g[seq_size:len(pred_final_train)+seq_size, :] = pred_final_train
# shift test predictions for plotting
testPredictPlot_g = np.empty_like(Dataset)
testPredictPlot_g[:, :] = np.nan
testPredictPlot_g[len(pred_final_train)+(seq_size*2)+1:len(Dataset)-1, :] = pred_final


# In[32]:


# plot baseline and predictions
plt.figure(figsize=(10,8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(scaler.inverse_transform(Dataset),color="magenta",label="True-result")
plt.plot(trainPredictPlot_g,color="cyan",label="Train-predicted-result")
plt.plot(testPredictPlot_g,color="black",label="Test-Predicted-result")
plt.legend()
plt.title("ENSEMBLE LEARNING PLOT")
plt.xlabel("Time Step(Months)")
plt.ylabel("Number of Air Passengers")
plt.grid(True)
plt.show()


# ## 6) Interpretation of the Results

# LSTM model has low accuracy compared to GRU
# Ensemble learning was developed by averaging both LSTM and GRU and it was found to have accuracy falling between LSTM and GRU

# In[33]:


import pickle


# In[34]:


pickle.dump(gru,open('gru.pkl','wb'))


# In[ ]:




