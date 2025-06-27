import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ✅ Load NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ✅ Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

# ✅ Streamlit UI
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")
st.title("📩 SMS Spam Detector")

# ✅ Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ✅ Input field
sms = st.text_area("📨 Enter the SMS message to classify:")

# ✅ Predict button
if st.button("Predict"):
    if not sms.strip():
        st.warning("⚠️ Please enter a message.")
    else:
        sms_cleaned = clean_text(sms)
        sms_vec = vectorizer.transform([sms_cleaned])
        prediction = model.predict(sms_vec)[0]
        if prediction == 1:
            st.error("🚨 This SMS is SPAM!")
        else:
            st.success("✅ This SMS is NOT spam.")
