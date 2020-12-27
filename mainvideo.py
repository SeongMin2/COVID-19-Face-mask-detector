from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import winsound

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('8LBMI2.h5')

# 실시간 웹캠 읽기
cap = cv2.VideoCapture(0)
i = 0

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(405, 405), mean=(104., 177., 123.))
    facenet.setInput(blob)
    dets = facenet.forward()


    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence < 0.5:
            continue

        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        face = img[y1:y2, x1:x2]
        face = face/256

        if (x2 >= w or y2 >= h):
            continue
        if (x1<=0 or y1<=0):
            continue

        face_input = cv2.resize(face,(200, 200))
        face_input = np.expand_dims(face_input, axis=0)
        face_input = np.array(face_input)

        modelpredict = model.predict(face_input)
        mask=modelpredict[0][0]
        nomask=modelpredict[0][1]

        if mask > nomask:
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
        else:
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100)
            #frequency = 2500  # Set Frequency To 2500 Hertz
            #duration = 1000  # Set Duration To 1000 ms == 1 second
            #winsound.Beep(frequency, duration)

        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=color, thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('masktest',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break