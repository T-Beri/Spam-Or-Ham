import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize stemmer
ps = PorterStemmer()

# Preprocess function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer & model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Streamlit UI
st.title("📩 SMS Spam Detection")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):   # 👈 only run when button clicked
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vectorInput = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vectorInput)[0]

    # 4. Display
    if result == 0:
        st.header("✅ Ham (Not Spam!)")
    else:
        st.header("🚨 Spam")
