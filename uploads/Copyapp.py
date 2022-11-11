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
app=Flask(__name__,template_folder='views')


# Model saved with Keras model.save()
MODEL_PATH = '/Users/vansh/Desktop/Project 3/LungCancerDetection.h5'

# Load your trained model
model =tf.keras.models.load_model('/Users/vansh/Desktop/Project 3/LungCancerDetection.h5')

# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(path, new_model):
   
    new_model = tf.keras.models.load_model('/Users/vansh/Desktop/Project 3/LungCancerDetection.h5')


    a  = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    b = cv2.resize(a, (256,256))


    y_pred = new_model.predict(np.expand_dims(b,0))
    index  = np.argmax(y_pred)

    if index == 0:
        print("Image belong to Bengin Cases")
    elif index == 1:
        print("Image belong to Malignant Cases")
    else:
        print("It looks normal")


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('/Users/vansh/Desktop/Project 3/Flaskapp/views/index.html')


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
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
                    # Convert to string
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)
