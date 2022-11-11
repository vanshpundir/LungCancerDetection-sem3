from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2


import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app=Flask(__name__)


# Model saved with Keras model.save()
MODEL_PATH = 'LungCancerDetection.h5'


model =tf.keras.models.load_model('/Users/vansh/Desktop/Project 3/LungCancerDetection.h5')


print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(path, new_model):
   
    new_model = tf.keras.models.load_model('LungCancerDetection.h5')


    a  = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    b = cv2.resize(a, (256,256))


    y_pred = new_model.predict(np.expand_dims(b,0))
    index  = np.argmax(y_pred)

    return y_pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = (model_predict(file_path, model))

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
                    # Convert to string
        if str(np.argmax(preds)) == "0":
            return "It belongs to Bengin Cases"
        elif str(np.argmax(preds)) == "1":
            return "It belongs to Malignant Cases"
        else:
            return "It belongs to Normal Cases"
    return None


if __name__ == '__main__':
    app.run(debug=True)
