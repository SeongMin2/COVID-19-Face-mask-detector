from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import winsound

# facenet : 얼굴을 찾는 모델
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
# model : 마스크 검출 모델
model = load_model('6LBMIv2-20.h5')

# 실시간 웹캠 읽기
cap = cv2.VideoCapture(0)
i = 0

while cap.isOpened():
    ret, img = cap.read()  # img는 3차원 넘파이 배열로 (높이, 넓이, 색상)형태임 색상은 3 (BGR) 로 되어 있을 것임
    if not ret:
        break

    # 이미지의 높이와 너비 추출
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(500, 500), mean=(104., 177., 123.))

    # facenet의 input으로 blob을 설정
    facenet.setInput(blob)
    # facenet 결과 추론, 얼굴 추출 결과가 dets의 저장
    dets = facenet.forward()

    # 마스크를 찾용했는지 확인
    for i in range(dets.shape[2]):   #dets.shape[2]

        # 검출한 결과가 신뢰도
        confidence = dets[0, 0, i, 2]
        # 신뢰도를 0.5로 임계치 지정
        if confidence < 0.5:
            continue

        # 바운딩 박스를 구함
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        # 원본 이미지에서 얼굴영역 추출
        face = img[y1:y2, x1:x2]
        # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) #  Gray할 때만 사용
        # face = np.expand_dims(face,axis=3)  #  Gray할 때만 사용
        face = face/256

        if (x2 >= w or y2 >= h):  # 얼굴인식중에 가끔 이상한 부분을 인식하는 경우를 무시하기 위한 코드
            continue
        if (x1<=0 or y1<=0):
            continue

        # 추출한 얼굴영역을 전처리
        face_input = cv2.resize(face,(200, 200))
        face_input = np.expand_dims(face_input, axis=0) # predict 할 때는 model input shape 보다 한 차원 더 높은 차원 넣어야함
        face_input = np.array(face_input)

        # 마스크 검출 모델로 결과값 return
        modelpredict = model.predict(face_input)
        mask=modelpredict[0][0]
        nomask=modelpredict[0][1]

        # 마스크를 꼈는지 안겼는지에 따라 라벨링해줌
        if mask > nomask:
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
        else:
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100)
            #frequency = 2500  # Set Frequency To 2500 Hertz
            #duration = 1000  # Set Duration To 1000 ms == 1 second
            #winsound.Beep(frequency, duration)

        # 화면에 얼굴부분과 마스크 유무를 출력해해줌
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=color, thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('masktest',img)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break