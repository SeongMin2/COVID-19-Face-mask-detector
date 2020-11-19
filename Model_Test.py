import os, re, glob
import cv2
import numpy as np
import shutil
from keras.models import load_model

def Dataization(img_path):
    img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #  Gray할 때만 사용
    return (img / 256)


src = []
name = []
test = []
image_dir = 'test_data/'
#image_dir = 'version1_data/v1_test_data/testtest/'
for file in os.listdir(image_dir):
    if (file.find('.jpg') is not -1):
        src.append(image_dir + file)  # dir랑 file명 합쳐서 저장(파일의 위치정보 있는 풀네임)
        name.append(file)             # 오직 file명만 저장
        test.append(Dataization(image_dir + file))   # 이미지들을 데이터화 시켜서 (숫자로) 사진마다 추가하여 저장

test = np.array(test)           # numpy 형태로 변환
#test = np.expand_dims(test,axis=3)  #  Gray할 때만 사용
print(test.shape)
model = load_model('6LBMIv2-20.h5')  # 테스트할 모델 로딩
predict = model.predict(test)   # 테스트할 데이터셋 입력
print(predict.shape) # predict 할 때는 model input shape 보다 한 차원 더 높은 차원 넣어야함
print("ImageName : , Predict : [mask, nomask]")
for i in range(len(test)):
    print(name[i] + " : , Predict : " + str(predict[i]))