import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import load_model

"""

0 :악성 기형성 콩팥증
1: 기저세포암
2: 양성 섬유종/각질종
3: 표피성 낭종/낭성내분비증
4: 흑색종양
5: 일반적인 피부
6: 혈관종

"""


model = load_model("C:/Users/T-user/Desktop/code/123/wookgid/model.h5")

img_path = "C:/Users/T-user/Desktop/code/123/wookgid/templates/bl.jpg"

# 이미지 로드 및 크기 조정
img = load_img(img_path, target_size=(64, 64))

# 이미지를 배열로 변환
img_array = img_to_array(img)

# 모델 입력 형식에 맞게 배열의 크기 재조정
input_array = img_array.reshape((1, 64, 64, 3))

# 모델 예측
predict = model.predict(input_array)

# 결과
predict_result = np.argmax(predict)

print(predict_result)
