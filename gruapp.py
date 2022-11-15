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
    html_time = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Air Passengers Prediction App </h2>
    </div>
    """
    timestamp = st.text_input("Timestamp","Type Here")
    safe_html="""  
    """
    st.header('Enter the timestamp:')
    Timestamp = st.number_input('Timestamp:', min_value=1, max_value=1000, value=1)
    result=""
    if st.button("Predict Passengers"):
        output = predict_Airpassengers(timestamp)
        st.success('The output is {}'.format(output))
if __name__=='__main__':
    main()


# In[ ]:




