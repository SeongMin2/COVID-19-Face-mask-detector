from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D, Conv2D
from keras.models import load_model
import os, re, glob
import cv2
import numpy as np

groups_folder_path = 'train_data_2(recommand)'
categories = ['mask','nomask']

num_classes = len(categories)


X = []
Y = []

for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]  # label=[0,0]  mask의 index가 0, nomask의 index가 1인데 일단 둘다 0으로 초기화
    label[idex] = 1                          # index가 0인경우 label=[1,0]  /  index가 1인경우 label=[0,1]
    image_dir = groups_folder_path+'/'+ categorie + '/'

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            #print(image_dir + filename)
            img = cv2.imread(image_dir + filename)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gray할 때만 사용
            X.append(img / 256) # 픽셀값은 0~255까지 존재, 학습시키에는 0~1값 사이가 적당하므로 256을 나눔
            Y.append(label)

Xtr = np.array(X)
#Xtr = np.expand_dims(Xtr,axis=3)  # Gray할 때만 사용
Ytr = np.array(Y)

X_train, Y_train = Xtr,Ytr
print(X_train.shape)
print(X_train.shape[1:])
###################################################################
#Validation 과정
'''
groups_folder_path2 = 'val_data_final'
categories = ['mask','nomask']

num_classes = len(categories)


Xv = []
Yv = []

for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path2+'/'+ categorie + '/'

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            #print(image_dir + filename)
            img = cv2.imread(image_dir + filename)
            Xv.append(img / 256) # 픽셀값은 0~255까지 존재, 학습시키에는 0~1값 사이가 적당하므로 256을 나눔
            Yv.append(label)

Xva = np.array(Xv)
Yva = np.array(Yv)

X_val, Y_val = Xva, Yva
print(X_val.shape)
print(X_val.shape[1:])
'''

model = Sequential() # Sequential 모델 object를 model이라는 변수에 넣음
model.add(Conv2D(16,3,3, border_mode='same', activation='relu',  # 필터수, 필터(커널)의 행,열, 출력이입력과 동일한크기로 padding설정, 활성화함수
                        input_shape=X_train.shape[1:]))   # (200, 200, 3), input_shape는 첫 레이어에만 넣으면 됨
model.add(MaxPooling2D(pool_size=(2, 2)))   # 수평, 수직 축소 비율-> 가로, 세로 절반으로 줄이겠다는 뜻
model.add(Dropout(0.25))   # dropout 설정

model.add(Convolution2D(20, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#Classifier
model.add(Flatten())  # 1차원으로 Flatten시킴
model.add(Dense(200, activation = 'relu'))  # 출력노드수, 활성화함수
# model.add(Dropout(0.5))  # classifier이므로 fully connected이니까 Dropout해주면 안됨
model.add(Dense(2,activation = 'softmax'))  # 출력노드수, (input_dim=입력뉴런수(맨처음입력층에서만 사용)), 활성화 함수
# Dense레이어는 입력과 출력을 모두 연결해주며 각각 연결해주는 가중치를 포함하고 있는 머신러닝의 기본층
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
hist=model.fit(X_train, Y_train, batch_size=40, nb_epoch=20)  #,validation_data=(X_val,Y_val) 필요시 사용

model.save('6LBMI-20.h5')
