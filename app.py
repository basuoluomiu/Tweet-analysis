from flask import Flask, render_template, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

model_name = "bert-base-uncased-finetuned-emotion"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = pipeline("text-classification", model=model_name, device=device)


def predict_sentiment(text):
    preds = model(text, top_k=None)
    preds = preds[0]
    return preds


def matching(predictions):
    emotions = ['悲伤', '高兴', '有爱', '愤怒', '恐惧', '惊喜']
    index = predictions.get('label')[6:]
    result = {'label': emotions[int(index)], 'score': predictions.get('score')}
    return result


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    predictions = predict_sentiment(text)
    predictions = matching(predictions)
    return jsonify(predictions)


if __name__ == '__main__':
    app.run(debug=True)
