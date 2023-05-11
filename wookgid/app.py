from flask import Flask,render_template,request
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import load_model

def predict(path):
    model=load_model('C:/Users/swl/Desktop/Code/wookgid/model.h5')

    img_path = path

    # 이미지 로드 및 크기 조정
    img = load_img(img_path, target_size=(64, 64))

    # 이미지를 배열로 변환
    img_array = img_to_array(img)

    # 모델 입력 형식에 맞게 배열의 크기 재조정
    input_array = img_array.reshape((1, 64, 64, 3))


    # 모델 예측
    predict= model.predict(input_array)
    
    # 이미지 경로
    predict_result = np.argmax (predict)
    return predict_result

app = Flask(__name__)


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
<html>
  <head>
    <title>이미지 업로드 예제</title>
  </head>
  <body>
    <h1>이미지 업로드 예제</h1>
    <input type="file" id="imageUpload">
    <br><br>
    <img id="preview">
    <script>
      // 이미지 업로드 폼 요소를 선택합니다.
      const inputElement = document.querySelector("#imageUpload");
      
      // 이미지 업로드 이벤트를 등록합니다.
      inputElement.addEventListener("change", handleFiles, false);

      function handleFiles() {
        // 업로드한 파일을 선택합니다.
        const fileList = this.files;
        const file = fileList[0];
        
        // 선택한 파일이 이미지인지 확인합니다.
        if (file.type.startsWith("image/")) {
          // 파일을 읽어들입니다.
          const reader = new FileReader();
          reader.onload = function() {
            // 읽어들인 파일을 이미지 요소에 표시합니다.
            const imgElement = document.querySelector("#preview");
            imgElement.src = reader.result;
          }
          reader.readAsDataURL(file);
        } else {
          alert("이미지 파일을 선택하세요.");
        }
      }
    </script>
  </body>
</html>
    '''
if __name__ == '__main__':
    app.debug = True
    app.run(port=5001)

