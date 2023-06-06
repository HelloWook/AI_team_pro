from flask import Flask, render_template, request
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


@app.route("/0")
def dis0():
    img_url = "/static/images/nv.jpg"
    return render_template("0.html", image_url=img_url)


@app.route("/1")
def dis1():
    img_url = "/static/images/흑색종jpg.jpg"
    return render_template("1.html", image_url=img_url)


@app.route("/2")
def dis2():
    img_url = "/static/images/양성 각화유사 병변.jpg"
    return render_template("2.html", image_url=img_url)


@app.route("/3")
def dis3():
    img_url = "/static/images/기저세포암.jpg"
    return render_template("3.html", image_url=img_url)


@app.route("/4")
def dis4():
    img_url = "/static/images/신성각화증.jpg"
    return render_template("4.html", image_url=img_url)


@app.route("/5")
def dis6():
    img_url = "/static/images/혈관병변.jpg"
    return render_template("5.html", image_url=img_url)


@app.route("/6")
def dis5():
    img_url = "/static/images/피부섬유종.jpg"
    return render_template("6.html", image_url=img_url)


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
    app.run(port=5000)
