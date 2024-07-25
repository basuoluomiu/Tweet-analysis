import requests
import random

url = "https://twitter154.p.rapidapi.com/search/search"

querystring = {"query": "life", "section": "top", "min_retweets": "1", "min_likes": "1", "limit": "5",
               "start_date": "2022-01-01", "language": "en"}

headers = {
    "x-rapidapi-key": "a13224d3f4msh913f30cb347563bp11fb67jsn51277cdf16e5",
    "x-rapidapi-host": "twitter154.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

data = response.json()['results']
new_data = []
for tweet in data:
    # print(tweet['user']['description'])
    new_data.append(tweet['user']['description'])

print(new_data[random.randint(0, len(new_data) - 1)])