import streamlit as st
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for impressive layout and category effects
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        color: #31333F;
        border-radius: 10px;
        border: 1px solid #ced4da;
    }
    .sentiment-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 24px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive {
        background: linear-gradient(135deg, #28a745, #218838);
    }
    .negative {
        background: linear-gradient(135deg, #dc3545, #c82333);
    }
    .neutral {
        background: linear-gradient(135deg, #ffc107, #e0a800);
        color: #212529;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        background-color: #007bff;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer with caching
@st.cache_resource
def load_models():
    model = None
    vectorizer = None
    
    if os.path.exists("sentiment.pkl") and os.path.exists("vector.pkl"):
        try:
            with open("sentiment.pkl", "rb") as f:
                model = pickle.load(f)
            with open("vector.pkl", "rb") as f:
                vectorizer = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model files: {e}")
            
    return model, vectorizer

model, vectorizer = load_models()

# App Header
st.title("📊 Advanced Sentiment Analysis")
st.markdown("Analyze the sentiment of your text instantly (Positive, Negative, or Neutral).")
st.markdown("---")

# Sidebar information
with st.sidebar:
    st.header("About App")
    st.info("This application uses a pre-trained Machine Learning model (`Naive Bayes`) combined with text vectorization to predict text sentiment accurately.")
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Type or paste your text in the box.")
    st.markdown("2. Click **Analyze Sentiment**.")
    st.markdown("3. View the categorized visual outcome.")

# Main Interface
user_input = st.text_area("Enter your text here:", placeholder="Type something like 'I love this product, it is amazing!'...", height=150)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    elif model is None or vectorizer is None:
        st.error("❌ Model or vectorizer files (`sentiment.pkl` / `vector.pkl`) could not be found or loaded correctly. Please check your directory.")
    else:
        try:
            # Transform input and predict
            transformed_input = vectorizer.transform([user_input])
            prediction = model.predict(transformed_input)[0]
            
            # Normalize prediction text for clean matching
            pred_lower = str(prediction).lower().strip()
            
            st.markdown("### Result Analysis:")
            
            # Display categorical styled card based on output
            if "pos" in pred_lower:
                st.markdown(
                    '<div class="sentiment-card positive">😊 Positive Sentiment Detected</div>', 
                    unsafe_allow_html=True
                )
            elif "neg" in pred_lower:
                st.markdown(
                    '<div class="sentiment-card negative">😞 Negative Sentiment Detected</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="sentiment-card neutral">😐 Neutral Sentiment Detected</div>', 
                    unsafe_allow_html=True
                )
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
