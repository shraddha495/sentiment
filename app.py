from flask import Flask, render_template_string, request
import pickle

app = Flask(_name_)

# Load the vectorizer and sentiment model
# Ensure 'vector.pkl' and 'sentiment.pkl' are in the same directory
with open('vector.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('sentiment.pkl', 'rb') as f:
    model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Sentiment Analysis Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --card-bg: rgba(255, 255, 255, 0.95);
            --text-color: #333333;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }
        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: var(--text-color);
            padding: 20px;
        }
        .container {
            background: var(--card-bg);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
            width: 100%;
            max-width: 600px;
            backdrop-filter: blur(10px);
            animation: fadeIn 0.8s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h2 {
            text-align: center;
            color: #4a3f8a;
            margin-bottom: 25px;
            font-weight: 700;
            font-size: 2rem;
        }
        textarea {
            width: 100%;
            height: 130px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 12px;
            font-size: 1rem;
            resize: none;
            transition: all 0.3s ease;
            outline: none;
        }
        textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
        }
        button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 14px;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        .result-box {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
            animation: popUp 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        @keyframes popUp {
            from { opacity: 0; transform: scale(0.9); }
            to { opacity: 1; transform: scale(1); }
        }
        .positive {
            background-color: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        .negative {
            background-color: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        .neutral {
            background-color: #fff3cd;
            color: #856404;
            border: 2px solid #ffeeba;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Sentiment Analyzer</h2>
        <form method="POST">
            <textarea name="text" placeholder="Type your text here to analyze sentiment..." required>{{ text if text else '' }}</textarea>
            <button type="submit">Analyze Sentiment</button>
        </form>
        {% if prediction %}
            <div class="result-box {{ prediction.lower() }}">
                Predicted Sentiment: <span>{{ prediction }}</span>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    text = ""
    if request.method == 'POST':
        text = request.form['text']
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)[0]
    return render_template_string(HTML_TEMPLATE, prediction=prediction, text=text)

if _name_ == '_main_':
    app.run(debug=True)
