#!/usr/bin/env python
# coding: utf-8

# ## Name:Benedict Odhiambo
# ## Reg no:21/08516
# ## Unit:MDA(Data Analytics and Knowledge Engineering)
# ## Assignment 4

# In[1]:


import streamlit as st
import pickle
import numpy as np
from keras.models import load_model
savedModel=load_model('gru_model.h5')
savedModel.summary()


# In[2]:


def welcome():
    return "Deployment of GRU Model"
def predict_Airpassengers(timestamp): 
    input=np.int(timestamp)
    prediction = savedModel.predict(input)
    print(Prediction)
    return Prediction
def main():
    st.title("Air Passengers Predictor")
    st.header('Enter the timestamp:')
    timestamp = st.number_input('timestamp:', min_value=1, max_value=1000, value=1)
    if st.button('Predict Passengers'):
        Passengers = predict_Airpassengers(timestamp)
        st.success(f'The predicted number of passengers is ${Prediction[0]:.2f}')
if __name__=='__main__': 
    main() 





