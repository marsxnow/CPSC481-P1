import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.notebook import tqdm
from collections import defaultdict

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input

dataset = "datasets/archive/UTKFace"
df = pd.DataFrame(columns=["img", "age", "gender", "ethnicity"])


for file_name in os.listdir(dataset):
    img_path = os.path.join(dataset, file_name)
    age_gender_ethnicity = file_name.split('_')
    age, gender, ethnicity = age_gender_ethnicity[0], age_gender_ethnicity[1], age_gender_ethnicity[2]
    try:
        df = pd.concat([df, pd.DataFrame([{"img": img_path, "age": int(age), "gender": int(gender), "ethnicity": int(ethnicity)}])], ignore_index=True)
    except:
        continue

