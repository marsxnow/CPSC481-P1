import os
from flask import Flask, request, jsonify, render_template, redirect
from flask_wtf import FlaskForm
from wtforms import FileField

from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img
app = Flask(__name__)

# Load model
model = load_model('models/model.keras')

# Check if file extension is valid
def allowed_file(filename: str) -> bool:
    file_extension = filename.split('.')[-1]

    return file_extension == "jpeg" or  file_extension == "jpg"
           

# Preprocess image
def preprocessImages(img):
    patterns = []
    img = load_img(img, color_mode='grayscale') #convert img to PIL format
    img = img.resize((128,128))
    img = np.array(img)
    patterns.append(img)

    patterns = np.array(patterns)
    patterns = patterns.reshape(len(patterns), 128, 128, 1)

    return patterns/255.0

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        print(request.files)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.save(os.path.join('uploads/',file.filename))
            return predict('uploads/' + file.filename)

    return render_template('failed.html')

@app.route('/predict', methods=['GET'])
def predict(image):
        image = preprocessImages(image)
        # Make prediction
        pred = model.predict(image[0].reshape(1, 128, 128, 1))
        # Format and return prediction
        gender_pred, age_pred = "%.f" % pred[0][0][0], "%.f" % pred[1][0][0]
        gender_pred = 'Male' if gender_pred == '0' else "Female"
        
        return render_template('results.html', gender=gender_pred, age=age_pred)

if __name__ == '__main__':
    app.run(debug=True)
