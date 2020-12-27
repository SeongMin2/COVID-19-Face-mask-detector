import cv2
import numpy as np
import os

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
with_dir = os.path.join('raw_data/with_mask2')
without_dir = os.path.join('raw_data/without_mask1')
print('total training withmask images:', len(os.listdir(with_dir)))
print('total training withoutmask images:', len(os.listdir(without_dir)))
withimgnum = len(os.listdir(with_dir))
withoutimgnum = len(os.listdir(without_dir))
with_files = os.listdir(with_dir)
without_files = os.listdir(without_dir)
print(withimgnum)
print(withoutimgnum)

for k in range(250,300): # Change range (the number of images)
    count=k
    img = cv2.imread('raw_data/with_mask2/' + with_files[k])
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(305,305), mean=(104., 177., 123.))
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

        if (x2 >= w or y2 >= h):
            continue

        face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (200, 200))

    file_name_path = 'train_data2/mask/trainnm' + str(count) + '.jpg'
    cv2.imwrite(file_name_path, face)
    print(count)

print("CopyComplete")
