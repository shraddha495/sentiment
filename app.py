import os
import joblib
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load model and vectorizer from the current directory
MODEL_PATH = "sentiment.pkl"
VECTORIZER_PATH = "vector.pkl"

# ==========================================
# LABEL FLIP TOGGLE:
# Set this to True if your model's outputs 
# are inverted (e.g. positive reads as negative)
# ==========================================
FLIP_LABELS = False 

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Sentiment Analysis</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            --card-bg: #ffffff;
            --text-color: #0f172a;
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
        }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--bg-gradient);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            box-sizing: border-box;
        }

        .container {
            background: var(--card-bg);
            padding: 35px 40px;
            border-radius: 20px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            width: 100%;
            max-width: 620px;
            box-sizing: border-box;
        }

        h2 {
            margin-top: 0;
            color: var(--primary-color);
            text-align: center;
            font-size: 28px;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .subtitle {
            text-align: center;
            color: #64748b;
            font-size: 14px;
            margin-bottom: 30px;
        }

        .form-group {
            margin-bottom: 22px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #334155;
            font-size: 14px;
        }

        textarea {
            width: 100%;
            padding: 14px;
            border: 2px solid #cbd5e1;
            border-radius: 12px;
            font-size: 15px;
            resize: vertical;
            min-height: 130px;
            box-sizing: border-box;
            font-family: inherit;
        }

        textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.12);
        }

        button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 14px 20px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            width: 100%;
            transition: background-color 0.2s;
        }

        button:hover {
            background-color: var(--primary-hover);
        }

        .result-card {
            margin-top: 30px;
            padding: 24px;
            border-radius: 14px;
            animation: fadeIn 0.4s ease;
        }

        .result-card.positive {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534;
        }

        .result-card.negative {
            background-color: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
        }

        .result-card.error {
            background-color: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }

        .result-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
            opacity: 0.8;
            margin-bottom: 6px;
        }

        .result-value {
            font-size: 22px;
            font-weight: 800;
            text-transform: capitalize;
        }

        .confidence-wrapper {
            margin-top: 12px;
            font-size: 13px;
        }

        .progress-bar-container {
            background-color: rgba(0, 0, 0, 0.08);
            border-radius: 6px;
            height: 8px;
            width: 100%;
            margin-top: 6px;
            overflow: hidden;
        }

        .progress-bar {
            height: 100%;
            border-radius: 6px;
        }

        .positive .progress-bar { background-color: #22c55e; }
        .negative .progress-bar { background-color: #ef4444; }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <div class="container">
        <h2>Sentiment Analyzer</h2>
        <div class="subtitle">AI-powered text emotion and sentiment evaluation</div>
        
        <form method="POST">
            <div class="form-group">
                <label for="text">Provide text input:</label>
                <textarea id="text" name="text" placeholder="Type or paste your review here..." required>{{ user_text if user_text else '' }}</textarea>
            </div>
            <button type="submit">Run Prediction</button>
        </form>

        {% if prediction %}
            <div class="result-card {{ card_type }}">
                <div class="result-title">Result Analysis</div>
                <div class="result-value">{{ prediction }}</div>
                
                {% if confidence is not none %}
                <div class="confidence-wrapper">
                    <div>Confidence Score: <strong>{{ "%.1f"|format(confidence * 100) }}%</strong></div>
                    <div class="progress-bar-container">
                        <div class="progress-bar" style="width: {{ confidence * 100 }}%;"></div>
                    </div>
                </div>
                {% endif %}
            </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    card_type = "positive"
    user_text = ""
    
    if request.method == "POST":
        user_text = request.form.get("text", "").strip()
        
        if not user_text:
            prediction = "Please enter valid text."
            card_type = "error"
        elif not model or not vectorizer:
            prediction = "Model files missing or not loaded correctly."
            card_type = "error"
        else:
            try:
                # Transform text and execute prediction
                transformed_text = vectorizer.transform([user_text])
                pred = model.predict(transformed_text)[0]
                
                # Normalize prediction to text format
                pred_str = str(pred).lower()
                is_positive = pred_str in ["positive", "pos", "1", "true", "happy"]
                
                # If your model labels are backwards, flip them via code config
                if FLIP_LABELS:
                    is_positive = not is_positive

                if is_positive:
                    prediction = "Positive"
                    card_type = "positive"
                else:
                    prediction = "Negative"
                    card_type = "negative"
                
                # Extract confidence probability if supported by model
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(transformed_text)[0]
                    confidence = float(max(probs))
                    
            except Exception as e:
                prediction = f"Processing Error: {str(e)}"
                card_type = "error"

    return render_template_string(
        HTML_TEMPLATE, 
        prediction=prediction, 
        confidence=confidence, 
        card_type=card_type, 
        user_text=user_text
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
