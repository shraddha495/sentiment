import os
from flask import Flask, render_template_string, request
from textblob import TextBlob

app = Flask(__name__)

# Modern HTML template with glowing star visual effects
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Star Sentiment Analyzer</title>
    <style>
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            box-sizing: border-box;
        }

        .container {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 40px;
            border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 550px;
            box-sizing: border-box;
            text-align: center;
        }

        h2 {
            color: #818cf8;
            font-size: 28px;
            font-weight: 800;
            margin-top: 0;
            margin-bottom: 8px;
        }

        .subtitle {
            color: #94a3b8;
            font-size: 14px;
            margin-bottom: 25px;
        }

        textarea {
            width: 100%;
            padding: 15px;
            background: rgba(15, 23, 42, 0.6);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: white;
            font-size: 15px;
            resize: vertical;
            min-height: 120px;
            box-sizing: border-box;
            font-family: inherit;
            outline: none;
            transition: border-color 0.2s;
        }

        textarea:focus {
            border-color: #818cf8;
            box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.15);
        }

        button {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            border: none;
            padding: 14px 20px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
            transition: transform 0.1s, opacity 0.2s;
        }

        button:hover {
            opacity: 0.95;
        }

        button:active {
            transform: scale(0.98);
        }

        .result-box {
            margin-top: 30px;
            padding: 25px;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .result-title {
            font-size: 12px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 700;
        }

        .stars {
            font-size: 34px;
            letter-spacing: 6px;
            margin: 15px 0;
            text-shadow: 0 0 20px rgba(250, 204, 21, 0.6);
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        .sentiment-label {
            font-size: 22px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .positive-text { color: #4ade80; }
        .negative-text { color: #f87171; }
        .neutral-text { color: #fbbf24; }

        .score-info {
            margin-top: 10px;
            font-size: 13px;
            color: #94a3b8;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes popIn {
            0% { transform: scale(0.5); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }
    </style>
</head>
<body>

    <div class="container">
        <h2>✨ Star Sentiment Analyzer</h2>
        <div class="subtitle">Accurate AI text emotion analysis with glowing star effects</div>
        
        <form method="POST">
            <textarea name="text" placeholder="Type or paste your text here (e.g., I like ice cream)..." required>{{ user_text if user_text else '' }}</textarea>
            <button type="submit">Analyze Sentiment</button>
        </form>

        {% if sentiment %}
            <div class="result-box">
                <div class="result-title">Rating Result</div>
                <div class="stars">{{ stars }}</div>
                <div class="sentiment-label {{ css_class }}">{{ sentiment }}</div>
                <div class="score-info">Polarity Score: {{ score }}</div>
            </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    stars = ""
    css_class = ""
    score = 0.0
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("text", "").strip()
        if user_text:
            # Analyze using TextBlob (No pickle files required)
            blob = TextBlob(user_text)
            score = round(blob.sentiment.polarity, 2)  # Score between -1.0 and 1.0
            
            if score > 0.05:
                sentiment = "Positive"
                css_class = "positive-text"
                stars = "⭐⭐⭐⭐⭐" if score > 0.4 else "⭐⭐⭐⭐☆"
            elif score < -0.05:
                sentiment = "Negative"
                css_class = "negative-text"
                stars = "⭐☆☆☆☆" if score < -0.4 else "⭐⭐☆☆☆"
            else:
                sentiment = "Neutral"
                css_class = "neutral-text"
                stars = "⭐⭐⭐☆☆"

    return render_template_string(
        HTML_TEMPLATE, 
        sentiment=sentiment, 
        stars=stars, 
        css_class=css_class, 
        score=score, 
        user_text=user_text
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
