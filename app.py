import streamlit as st
import numpy as np
import ktrain
from ktrain import text
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# @st.cache(allow_output_mutation=True)
# def get_model():
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertForSequenceClassification.from_pretrained("pnichite/YTFineTuneBert")
#     return tokenizer,model



# Load the saved BERT model
loaded_model = ktrain.load_predictor('path_to_saved_model')

# Create a Streamlit application for text classification
st.title('Text Classification App')
user_input = st.text_input('Enter the text to classify')
if st.button('Classify'):
    prediction = loaded_model.predict(user_input)
    st.write('The text is classified as:', prediction)

# tokenizer,model = get_model()

# user_input = st.text_area('Enter Text to Analyze')
# button = st.button("Analyze")

# d = {
    
#   1:'Toxic',
#   0:'Non Toxic'
# }

# if user_input and button :
#     test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
#     # test_sample
#     output = model(**test_sample)
#     st.write("Logits: ",output.logits)
#     y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
#     st.write("Prediction: ",d[y_pred[0]])




















# import streamlit as st
# import sklearn
# from sklearn.linear_model import LogisticRegression  
# import pickle
# import pandas
# import joblib

# # loading the trained model
# model = pickle.load(open('Pickle_RL_Model.pkl', 'rb'))

# # create title
# st.title('Truth Seeker')

# message = st.text_input('Enter an article')

# submit = st.button('Predict')

# if submit:
#     prediction = model.predict([message])

#     # print(prediction)
#     # st.write(prediction)
    
#     if prediction[0] == 'spam':
#         st.warning('This Article is Propagandistic')
#     else:
#         st.success('This article is Non-Propagandistic')


# st.balloons()
