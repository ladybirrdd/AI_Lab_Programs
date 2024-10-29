
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("playsheet_dataset.csv")


outlook_encoder = LabelEncoder()
temp_encoder = LabelEncoder()
humidity_encoder = LabelEncoder()
windy_encoder = LabelEncoder()


inputs = df.drop("Play", axis="columns")
inputs["outlook_n"] = outlook_encoder.fit_transform(inputs["Outlook"])
inputs["temp_n"] = temp_encoder.fit_transform(inputs["Temp"])
inputs["humidity_n"] = humidity_encoder.fit_transform(inputs["Humidity"])
inputs["windy_n"] = windy_encoder.fit_transform(inputs["Windy"])


inputs_n = inputs.drop(["Outlook", "Temp", "Humidity", "Windy"], axis="columns")

target = df["Play"]

# Apply Gaussian Naive Bayes
classifier = GaussianNB()
classifier.fit(inputs_n, target)


print("GaussianNB Model Trained")

predicted = classifier.predict(inputs_n)
print(predicted)


inputs


inputs_n
