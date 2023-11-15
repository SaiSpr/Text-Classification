import streamlit as st
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification

st.title("Truth Seeker App")

# option = st.selectbox(
#     "Select an Option",
#     [
#         "Classify Text",
#         "Question Answering",
#         "Text Generation",
#         "Named Entity Recognition",
#         "Summarization",
#         "Translation",
#     ],
# )

# if option == "Classify Text":
user_input = st.text_area('Enter Text to Analyze')

if user_input is not None:
    if st.button("Analyse"):
        classifier = pipeline("sentiment-analysis")
        prediction  = classifier(text)
        # class_name = "Propagandistic" if prediction == 1 else "Non-Propagandistic"
        # st.subheader("Result:")
        # st.info("The article is "+ result + ".")
        st.write(prediction)
        st.write(prediction[0])

# elif option == "Question Answering":
#     q_a = pipeline("question-answering")
#     context = st.text_area(label="Enter context")
#     question = st.text_area(label="Enter question")
#     if context and question:
#         answer = q_a({"question": question, "context": context})
#         st.write(answer)
# elif option == "Text Generation":
#     text = st.text_area(label="Enter text")
#     if text:
#         text_generator = pipeline("text-generation")
#         answer = text_generator(text)
#         st.write(answer)
# elif option == "Named Entity Recognition":
#     text = st.text_area(label="Enter text")
#     if text:
#         ner = pipeline("ner")
#         answer = ner(text)
#         st.write(answer)
# elif option == "Summarization":
#     summarizer = pipeline("summarization")
#     article = st.text_area(label="Paste Article")
#     if article:
#         summary = summarizer(article, max_length=400, min_length=30)
#         st.write(summary)
# elif option == "Translation":
#     translator = pipeline("translation_en_to_de")
#     text = st.text_area(label="Enter text")
#     if text:
#         translation = translator(text)
#         st.write(translation)





@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("pnichite/YTFineTuneBert")
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

d = {
    
  1:'Toxic',
  0:'Non Toxic'
}

if user_input and button :
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",d[y_pred[0]])



















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
