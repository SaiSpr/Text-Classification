import streamlit as st
import pickle

# loading the trained model
#model = pickle.load(open('model.pkl', 'rb'))

# create title
st.title('Predicting if message is spam or not')

message = st.text_input('Enter a message')

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
