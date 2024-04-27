import os 
from PIL import Image
import pandas as pd
import numpy as np

from keras.preprocessing.image import load_img
from keras.models import load_model
import tensorflow as tf

def preprocessImages(file):
    patterns = []
    file = load_img(file, color_mode='grayscale') #convert img to PIL format
    file = file.resize((128,128))
    file = np.array(file)
    patterns.append(file)

    patterns = np.array(patterns)
    patterns = patterns.reshape(len(patterns), 128, 128, 1)
    return patterns

df = pd.read_pickle("df.pkl")
df = df.sample(n=14400, random_state=42)

model = load_model("model.keras")

temp_img = os.path.join("", "me.jpg")
temp_img = preprocessImages(temp_img)
temp_img = temp_img/255.0

# print("Original Gender:", y_gender[0], "Original Age:", y_age[0])
pred = model.predict(temp_img[0].reshape(1, 128, 128, 1))
gender_pred, age_pred = "%.f" % pred[0][0][0], "%.f" % pred[1][0][0]
print(gender_pred, age_pred)

