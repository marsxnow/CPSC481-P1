import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.notebook import tqdm
from PIL import Image

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.utils import plot_model


def get_patterns(images_dataframe):
    patterns = []
    for img in images_dataframe:
        img = load_img(img, color_mode="grayscale")  # convert img to PIL format
        img = img.resize((128, 128))
        img = np.array(img)
        patterns.append(img)

    patterns = np.array(patterns)
    patterns = patterns.reshape(len(patterns), 128, 128, 1)
    return patterns


dataset = "datasets/archive/myImg"
df = pd.DataFrame(columns=["img", "age", "gender", "ethnicity"])

for file_name in os.listdir(dataset):
    img_path = os.path.join(dataset, file_name)
    age_gender_ethnicity = file_name.split("_")
    # print(file_name)
    try:
        age, gender, ethnicity = (
            age_gender_ethnicity[0],
            age_gender_ethnicity[1],
            age_gender_ethnicity[2],
        )
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "img": img_path,
                            "age": int(age),
                            "gender": int(gender),
                            "ethnicity": int(ethnicity),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    except:
        continue

# df.to_pickle("df_subset.pkl")

# df = pd.read_pickle("df_subset.pkl")
# df = df.sample(n=14400, random_state=42)

X = get_patterns(df["img"])  # numpy array
X = X / 255.0

y_age = np.array(df["age"])
y_gender = np.array(df["gender"])

input_shape = (128, 128, 1)

inputs = Input((input_shape))
# convolutional layers
conv_1 = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation="relu")(maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation="relu")(maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation="relu")(maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

flatten = Flatten()(maxp_4)

dense_1 = Dense(256, activation="relu")(flatten)
dense_2 = Dense(256, activation="relu")(flatten)

dropout_1 = Dropout(0.4)(dense_1)
dropout_2 = Dropout(0.4)(dense_2)

output_1 = Dense(
    1, activation="sigmoid", name="gender_out"
)(
    dropout_1
)  # use sigmoid function(range from [0,1]) since gender can only be 0(male) or 1(female)
output_2 = Dense(1, activation="relu", name="age_out")(dropout_2)


# model = Model(inputs=[inputs], outputs=[output_1, output_2])
# model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy', 'mae'])
model = load_model("model.keras")
# plot_model(model, to_file='model.png')


X = tf.convert_to_tensor(X, dtype=tf.float32)
y_gender = tf.convert_to_tensor(y_gender, dtype=tf.float32)
y_age = tf.convert_to_tensor(y_age, dtype=tf.float32)

train = model.fit(
    x=X,
    y=[y_gender, y_age],
    batch_size=32,
    epochs=200,
    shuffle=True,
    validation_split=0.2,
)
model.save("model.keras")
