import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from keras.layers.normalization import BatchNormalization

from PIL import Image

import os
from flask import Flask, url_for, render_template, request, redirect, send_from_directory, flash, session
from werkzeug.utils import secure_filename
import cfg

# Definitely we can load the model once and use it
# but it was for some reason giving some error
# so for now it is sufficient

app = Flask(__name__)
app.secret_key = "masterCS"
app.config['UPLOAD_FOLDER'] = cfg.UPLOAD_IMAGES_PATH

def load_model():
    #Use VGG16 imagenet weights as base model
    input_tensor = Input(shape=cfg.img_dim)
    base_model = VGG16(include_top=False,
                           weights='imagenet',
                           input_shape=cfg.img_dim)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Flatten()(x)
    output = Dense(cfg.num_classes, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    #loading the weights that we trained
    model.load_weights('notebooks/weights/weights.best.hdf5')

    print("Model loaded!")
    return model



def preprocess_image(filename):
    img = Image.open(filename)
    img.thumbnail((128, 128))

    # Convert to RGB and normalize
    img_array = np.asarray(img.convert("RGB"), dtype=np.float32)

    img_array = img_array[:, :, ::-1]
    # Zero-center by mean pixel
    img_array[:, :, 0] -= 103.939
    img_array[:, :, 1] -= 116.779
    img_array[:, :, 2] -= 123.68

    img_array = img_array / 255
    img_array = np.expand_dims(img_array, axis=0)
    print("Input preprocessed!")
    return img_array

def predictClass(img_array):
    model = load_model()
    yhat = model.predict(img_array)
    # This We use 0 because we are only predicting one image
    # This could br easily extended to multiple images
    labels = [cfg.ymap[i] for i, value in enumerate(yhat[0]) if value > cfg.thresholds[i]]
    print(labels)
    return labels

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
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
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['filename'] = filename
            return redirect(url_for('predict'))
    return render_template('home.html')

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/predict')
def predict():
    filename = session['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    img_array = preprocess_image(filepath)
    labels = predictClass(img_array)

    return render_template('predict.html', image=filename, labels=labels)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
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
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['filename'] = filename
            return redirect(url_for('predict'))

    return render_template('input.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in cfg.ALLOWED_EXTENSIONS



if __name__ == '__main__':
    app.run(port=3303, debug=True)
