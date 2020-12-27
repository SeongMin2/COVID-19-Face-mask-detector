import os, re, glob
import cv2
import numpy as np
import shutil
from keras.models import load_model

def Dataization(img_path):
    img = cv2.imread(img_path)
    return (img / 256)


src = []
name = []
test = []
image_dir = 'test_data/'
for file in os.listdir(image_dir):
    if (file.find('.jpg') is not -1):
        src.append(image_dir + file)
        name.append(file)
        test.append(Dataization(image_dir + file))

test = np.array(test)
print(test.shape)
model = load_model('6LBMIv2-20.h5')
predict = model.predict(test)
print(predict.shape)
print("ImageName : , Predict : [mask, nomask]")
for i in range(len(test)):
    print(name[i] + " : , Predict : " + str(predict[i]))