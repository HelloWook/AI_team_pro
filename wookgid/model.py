import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import load_model


lesion_type_dict = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions ",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}

model = load_model("C:/Users/T-user/Desktop/code/123/wookgid/model.h5")

img_path = "C:/Users/T-user/Desktop/code/123/wookgid/templates/diseaseimage/신성각화증.jpg"

# 이미지 로드 및 크기 조정
img = load_img(img_path, target_size=(64, 64))

# 이미지를 배열로 변환
img_array = img_to_array(img)

# 모델 입력 형식에 맞게 배열의 크기 재조정
input_array = img_array.reshape((1, 64, 64, 3))

# 모델 예측
predict = model.predict(input_array)
probabilities = tf.nn.softmax(predict).numpy()

# 결과
predict_result_index = np.argmax(predict)
predict_class = list(lesion_type_dict.keys())[predict_result_index]
predict_label = lesion_type_dict[predict_class]

print("Predicted class:", predict_class)
print("Predicted label:", predict_label)
