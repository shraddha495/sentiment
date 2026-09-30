import os
import joblib
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load model and vectorizer from the current directory
MODEL_PATH = "sentiment.pkl"
VECTORIZER_PATH = "vector.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

# HTML Template with modern embedded CSS and categorical card effects
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis Dashboard</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            --card-bg: #ffffff;
            --text-color: #1f2937;
            --primary-color: #6366f1;
            --primary-hover: #4f46e5;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-gradient);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }

        .container {
            background: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            width: 100%;
            max-width: 600px;
            box-sizing: border-box;
            transition: transform 0.3s ease;
        }

        .container:hover {
            transform: translateY(-3px);
        }

        h2 {
            margin-top: 0;
            color: var(--primary-color);
            text-align: center;
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 25px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #374151;
        }

        textarea {
            width: 100%;
            padding: 14px;
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            font-size: 15px;
            resize: vertical;
            min-height: 120px;
            box-sizing: border-box;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
        }

        button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 14px 20px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            width: 100%;
            transition: background-color 0.2s, transform 0.1s;
        }

        button:hover {
            background-color: var(--primary-hover);
        }

        button:active {
            transform: scale(0.98);
        }

        /* Categorical Form & Layout Effects */
        .result-card {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .result-card.positive {
            background-color: #ecfdf5;
            color: #065f46;
            border: 1px solid #34d399;
        }

        .result-card.negative {
            background-color: #fef2f2;
            color: #991b1b;
            border: 1px solid #f87171;
        }

        .result-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-bottom: 6px;
            font-weight: 700;
            opacity: 0.8;
        }

        .result-value {
            font-size: 26px;
            font-weight: 800;
            text-transform: capitalize;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <div class="container">
        <h2>Sentiment Analyzer</h2>
        <div class="subtitle">Analyze text sentiment using your Naive Bayes classifier</div>
        
        <form method="POST">
            <div class="form-group">
                <label for="text">Enter Review or Sentence:</label>
                <textarea id="text" name="text" placeholder="Type or paste your text here..." required>{{ user_text if user_text else '' }}</textarea>
            </div>
            <button type="submit">Predict Sentiment</button>
        </form>

        {% if prediction %}
            <div class="result-card {{ 'positive' if prediction.lower() in ['positive', 'pos', '1'] else 'negative' }}">
                <div class="result-title">Predicted Category</div>
                <div class="result-value">{{ prediction }}</div>
            </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_text = ""
    if request.method == "POST":
        user_text = request.form.get("text", "")
        if user_text and model and vectorizer:
            try:
                # Vectorize input and predict
                transformed_text = vectorizer.transform([user_text])
                pred = model.predict(transformed_text)[0]
                prediction = str(pred)
            except Exception as e:
                prediction = f"Error: {str(e)}"
        elif not model or not vectorizer:
            prediction = "Model files missing or not loaded correctly."

    return render_template_string(HTML_TEMPLATE, prediction=prediction, user_text=user_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
