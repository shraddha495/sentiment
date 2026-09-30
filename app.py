import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI Sentiment Analysis Dashboard",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling & Visual Effects
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 10px;
    }
    .sentiment-card {
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .positive { background: linear-gradient(135deg, #28a745, #20c997); }
    .negative { background: linear-gradient(135deg, #dc3545, #f86f70); }
    .neutral { background: linear-gradient(135deg, #ffc107, #ff9800); color: #333 !important; }
    
    .stars {
        font-size: 30px;
        color: #ffD700;
        text-align: center;
        margin-top: 10px;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Load Vectorizer and Model with Caching
@st.cache_resource
def load_models():
    try:
        with open('vector.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('sentiment.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

vectorizer, model = load_models()

# App Header
st.title("🌟 AI Sentiment Analysis Hub")
st.markdown("Analyze customer reviews, feedback, or text instantly using your trained Naive Bayes machine learning model.")
st.markdown("---")

# Sidebar Information
with st.sidebar:
    st.header("About App")
    st.info("This application evaluates text input and categorizes it into **Positive**, **Neutral**, or **Negative** sentiments with immersive design elements.")
    st.markdown("---")
    st.markdown("### Model Status")
    if vectorizer and model:
        st.success("Models Loaded Successfully! ✅")
    else:
        st.error("Model Files Missing or Corrupted ❌")

# Main Input Section
st.subheader("✍️ Enter text for analysis:")
user_input = st.text_area("", placeholder="Type your review or sentence here...", height=120)

if st.button("🚀 Analyze Sentiment", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    elif vectorizer is None or model is None:
        st.error("⚠️ Models are not loaded properly. Check your .pkl files.")
    else:
        try:
            # Transform and Predict
            transformed_input = vectorizer.transform([user_input])
            prediction = model.predict(transformed_input)
            
            # Normalize prediction string
            sentiment = str(prediction[0]).strip().lower()
            
            # Display Result based on Sentiment Category
            if "pos" in sentiment:
                st.markdown("""
                    <div class="sentiment-card positive">
                        😊 Positive Sentiment Detected!
                    </div>
                    <div class="stars">⭐⭐⭐⭐⭐</div>
                """, unsafe_allow_html=True)
            elif "neg" in sentiment:
                st.markdown("""
                    <div class="sentiment-card negative">
                        😞 Negative Sentiment Detected!
                    </div>
                    <div class="stars">⭐☆☆☆☆</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="sentiment-card neutral">
                        😐 Neutral Sentiment Detected!
                    </div>
                    <div class="stars">⭐⭐⭐☆☆</div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Powered by Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)
