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
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path+'/'+ categorie + '/'

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            img = cv2.imread(image_dir + filename)
            X.append(img / 256)
            Y.append(label)

Xtr = np.array(X)
Ytr = np.array(Y)

X_train, Y_train = Xtr,Ytr
print(X_train.shape)
print(X_train.shape[1:])

model = Sequential()
model.add(Conv2D(16,3,3, border_mode='same', activation='relu',
                        input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

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
model.add(Flatten())
model.add(Dense(200, activation = 'relu'))
model.add(Dense(2,activation = 'softmax'))

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
hist=model.fit(X_train, Y_train, batch_size=40, nb_epoch=20)

model.save('8LBMI3.h5')
