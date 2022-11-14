#!/usr/bin/env python
# coding: utf-8

# ## Name:Benedict Odhiambo
# ## Reg no:21/08516
# ## Unit:MDA(Data Analytics and Knowledge Engineering)
# ## Assignment 4

# In[1]:


get_ipython().system('pip install streamlit')


# In[2]:


import streamlit
import pickle
import numpy as np


# In[3]:


model = pickle.load(open('./gru.pkl','rb'))
def predict_Airpassengers(Timestamp): 
    prediction = model.predict('Timestamp')
    return prediction


# In[4]:


from typing_extensions import TypeGuard


# In[5]:


import streamlit as st
st.title('Air Passengers Predictor')
st.image('D:\Ben Important\Master Data Analytics\MSC 2.1 Notes\Data Analytics and Knowledge Engineering\Kenya-Airways-celebrates-inaugural-U.S.-flight.png')
st.header('Enter the Timestamp:')
Timestamp = st.number_input('Timestamp:', min_value=1, max_value=1000, value=1)
if st.button('Predict Passengers'):
    passengers = predict(Timestamp)
    st.success(f'The predicted number of passengers is {passengers[0]:.2f} persons')

