from flask import Flask, render_template, request, url_for
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import load_model

app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["image"]
        image_path = "./images/" + file.filename
        file.save(image_path)
        model = load_model("C:/Users/T-user/Desktop/code/123/wookgid/model.h5")
        img = load_img(image_path, target_size=(64, 64))
        img_array = img_to_array(img)
        input_array = img_array.reshape((1, 64, 64, 3))
        predict = model.predict(input_array)
        predict_result = np.argmax(predict)
    return render_template("index.html", prediciton=predict_result)


if __name__ == "__main__":
    app.run()
