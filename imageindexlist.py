import cv2
import numpy as np
import os

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
with_dir = os.path.join('raw_data/with_mask2')
without_dir = os.path.join('raw_data/without_mask2')
print('total training withmask images:', len(os.listdir(with_dir)))
print('total training withoutmask images:', len(os.listdir(without_dir)))
withimgnum = len(os.listdir(with_dir))
withoutimgnum = len(os.listdir(without_dir))
with_files = os.listdir(with_dir)
without_files = os.listdir(without_dir)

for i in enumerate(with_files):
    print(i)