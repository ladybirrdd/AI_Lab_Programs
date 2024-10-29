import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4") 


weather_rules = {
    "sunny,warm": "It is likely to be a nice day. Enjoy outdoor activities!",
    "rainy,cold": "Expect rain and chilly weather. Carry an umbrella!",
    "sunny,cool": "A cool but sunny day is expected. Perfect for a walk!",
    "cloudy,warm": "It might be cloudy, but the weather will be warm.",
    "stormy": "Severe weather warning! Stay indoors!",
    "snowy": "Expect snowfall. Drive safely and stay warm!",
}


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_input(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]
    return filtered_tokens


def forecast_weather(conditions):
    conditions_joined = ",".join(conditions)
    conditions_sorted = ",".join(sorted(conditions))
    forecast = weather_rules.get(
        conditions_joined,
        weather_rules.get(
            conditions_sorted,
            "Weather conditions unclear. Please provide more details.",
        ),
    )
    return forecast

user_input = input("Describe the weather conditions (e.g., 'sunny, warm'): ")

user_input = user_input.replace("'", "").replace('"', "").strip()

user_conditions = preprocess_input(user_input)

print("Processed Conditions:", user_conditions)
print("Weather Forecast:", forecast_weather(user_conditions))
