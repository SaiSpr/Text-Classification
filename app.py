import streamlit as st
import sklearn
from sklearn.linear_model import LogisticRegression  
import pickle
import pandas
import joblib

# loading the trained model
model = pickle.load(open('Pickle_RL_Model.pkl', 'rb'))

# create title
st.title('Truth Seeker')

message = st.text_input('Enter an article')

submit = st.button('Predict')

if submit:
    prediction = model.predict([message])

    # print(prediction)
    # st.write(prediction)
    
    if prediction[0] == 'spam':
        st.warning('This Article is Propagandistic')
    else:
        st.success('This article is Non-Propagandistic')


st.balloons()
