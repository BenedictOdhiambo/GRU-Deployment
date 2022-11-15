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
model = open('./model.pkl','rb')


# In[2]:


def welcome():
    return "Deployment of GRU Model"
def predict_Airpassengers(timestamp): 
    input=np.int(timestamp)
    prediction = model.predict(input)
    print(Prediction)
    return Prediction


# In[3]:


def main():
    st.title("Air Passengers Predictor")
    st.header('Enter the timestamp:')
    timestamp = st.number_input('timestamp:', min_value=1, max_value=1000, value=1)
    output=""
    if st.button("Predict Passengers"):
        output = predict_Airpassengers(timestamp)
        st.success('The output is {}'.format(output))
if __name__=='__main__':
    main()


# In[ ]:




