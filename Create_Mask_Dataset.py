import dlib
import cv2
import numpy as np
from PIL import Image, ImageFile
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

without_dir = os.path.join('without_mask2')
print('total training withoutmask images:', len(os.listdir(without_dir)))
withoutimgnum = len(os.listdir(without_dir))
without_files = os.listdir(without_dir)

for k in range(500,551):
    count = k
    img = cv2.imread('without_mask2/' + without_files[k], 1)

    rows, cols = img.shape[:2]
    rects = detector(img, 1)
    for i, rect in enumerate(rects):
        shape = predictor(img, rect)
        for j in range(68):
            x, y = shape.part(j).x, shape.part(j).y

            color = (0, 255, 0)
            if (j == 3):
                color = (0, 0, 255)
                left = np.array([x, y])
            elif (j == 8):
                color = (0, 0, 255)
                chin = np.array([x, y])
            elif (j == 13):
                color = (0, 0, 255)
                right = np.array([x, y])
            elif (j == 29):
                color = (0, 0, 255)
                nose = np.array([x, y])
            cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, color)


    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


    img2 = Image.open('without_mask2/' + without_files[k])
    mask_img = Image.open('black-mask2.png')
    width = mask_img.width
    height = mask_img.height
    width_ratio = 1.2
    new_height = int(np.linalg.norm(left - right))

    # left
    mask_left_img = mask_img.crop((0, 0, width // 2, height))
    mask_left_width = get_distance_from_point_to_line(left, nose, chin)
    mask_left_width = int(mask_left_width * width_ratio)
    mask_left_img = mask_left_img.resize((mask_left_width, new_height))

    # right
    mask_right_img = mask_img.crop((width // 2, 0, width, height))
    mask_right_width = get_distance_from_point_to_line(right, nose, chin)
    mask_right_width = int(mask_right_width * width_ratio)
    mask_right_img = mask_right_img.resize((mask_right_width, new_height))

    # merge mask
    size = (mask_left_img.width + mask_right_img.width, new_height)
    mask_img = Image.new('RGBA', size)
    mask_img.paste(mask_left_img, (0, 0), mask_left_img)
    mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

    # rotate mask
    angle = np.arctan2(chin[1] - nose[1], chin[0] - nose[0])
    rotated_mask_img = mask_img.rotate(angle, expand=True)

    # calculate mask location
    center_x = (nose[0] + chin[0]) // 2
    center_y = (nose[1] + chin[1]) // 2

    offset = mask_img.width // 2 - mask_left_img.width
    radian = angle * np.pi / 180
    box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
    box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

    img2.paste(rotated_mask_img, (box_x, box_y), rotated_mask_img)

    file_name_path = 'test_data_final/bm/testbm' + str(count) + '.jpg' # your own repository path
    img2.save(file_name_path)
    print(count)