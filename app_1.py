import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Define the model architecture
model = Sequential([
    # Assuming embedding_layer is defined elsewhere or loaded
    Bidirectional(LSTM(64, return_sequences=True)),  # Adjust LSTM size to match your embedding dimension
    Bidirectional(LSTM(64)),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile Model 1
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load Model 1 weights

model = tf.keras.models.load_model('sentiment_model.h5')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Set the max length (as used in training)
max_len = 100

# Function to preprocess the input
def preprocess_input(review_text):
    sequence = tokenizer.texts_to_sequences([review_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    return padded_sequence

# Prediction function
def predict_review(review_text):
    padded_sequence = preprocess_input(review_text)
    prediction = model.predict(padded_sequence)
    st.write(f"Raw prediction value: {prediction[0][0]}")  # Debugging: print raw prediction
    
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment

# Streamlit App
def main():
    # App Title and Design
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .header {
            font-size: 60px;
            color: #FFA07A;
            font-family: 'Courier New', Courier, monospace;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
        }
        .stars {
            color: #FFD700;
            text-align: center;
            font-size: 30px;
        }
        .footer {
            font-size: 15px;
            color: #888;
            text-align: center;
            margin-top: 50px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="header">Clothing Review Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="stars">⭐⭐⭐⭐⭐</div>', unsafe_allow_html=True)

    # Input Box for Review Text
    review_text = st.text_area("Enter your review:", placeholder="Write your review here...", height=200)

    # Prediction Button
    if st.button("Predict Sentiment"):
        if review_text:
            sentiment = predict_review(review_text)
            st.success(f"Predicted Sentiment: **{sentiment}**")
        else:
            st.warning("Please enter a review to predict sentiment.")

    # Footer with your name
    st.markdown('<div class="footer">Developed by Sawaira Waheed</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
