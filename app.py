from flask import Flask, render_template, request, jsonify
import torch
from transformers import pipeline
import requests
import random

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


@app.route('/random_tweet', methods=['GET'])
def random_tweet():
    url = "https://twitter154.p.rapidapi.com/search/search"
    querystring = {
        "query": random.choice(
            ["life", "work", "love", "friendship", "school", "journey", "emotion", "mood", "feel", "memories",
             "recall", "Voice", "share", "blessing"]),
        "section": "top",
        "min_retweets": "1",
        "min_likes": "1",
        "limit": "5",
        "start_date": "2022-01-01",
        "language": "en"
    }
    headers = {
        "x-rapidapi-key": "a13224d3f4msh913f30cb347563bp11fb67jsn51277cdf16e5",
        "x-rapidapi-host": "twitter154.p.rapidapi.com"
    }

    tweet = None
    while not tweet:
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json().get('results', None)
        if data:
            new_data = [tweet['user']['description'] for tweet in data]
            tweet = new_data[random.randint(0, len(new_data) - 1)] if new_data else None

    return jsonify({'tweet': tweet})


if __name__ == '__main__':
    app.run(debug=True)
