import dlib
# 현재 사용중인 anaconda 가상환경에 dlib 설치함
# anaconda prompt 명령어 창에서
# conda install -c conda-forge dlib
import cv2
import numpy as np
from PIL import Image, ImageFile
# Pillow가 이미지 편집에 용이함, crop, paste 등 사용가능
import os

detector = dlib.get_frontal_face_detector() # 얼굴 rectangle detecter
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 외부 facial landmark 사용

without_dir = os.path.join('without_mask2')
print('total training withoutmask images:', len(os.listdir(without_dir)))
withoutimgnum = len(os.listdir(without_dir))
without_files = os.listdir(without_dir)

for k in range(500,551):
    count = k
    img = cv2.imread('without_mask2/' + without_files[k], 1)  # 기본 인물 사진

    rows, cols = img.shape[:2]   # 인물 사진의 가로와 세로

    #r = 500. / img.shape[1]            # 500 부분 숫자가 클수록 이미지가 커지지만 화질은 안좋아짐
    #dim = (500, int(img.shape[0] * r)) # 밑에 detector, predictor에 매개변수로 넣어줄 때 그냥 opencv resize뜨면 오류가 떠서 이런식으로 resisze 함
    # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)  # 이미지 resize
    rects = detector(img, 1)     # 얼굴 detector
    for i, rect in enumerate(rects):
        shape = predictor(img, rect)
        for j in range(68):     # 68개의 점을 찍어줌
            x, y = shape.part(j).x, shape.part(j).y   # 얼굴의 점의 좌표 x, y
            # cv2.circle(img, (x, y), 1, (0, 255, 0), -1) # landmark를 점으로 찍을 때의 코드
            color = (0, 255, 0)  # 초록
            if (j == 3):     # 얼굴의 왼쪽 턱 일 경우
                color = (0, 0, 255)
                left = np.array([x, y])
            elif (j == 8):   # 얼굴의 밑 턱일 경우
                color = (0, 0, 255)
                chin = np.array([x, y])
            elif (j == 13):  # 얼굴의 오른쪽 턱일 경우
                color = (0, 0, 255)
                right = np.array([x, y])
            elif (j == 29):  # 얼굴의 코 중앙일 경우
                color = (0, 0, 255)
                nose = np.array([x, y])
            cv2.putText(img, str(j), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, color) # landmark를 숫자로 표현 0~68


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
    new_height = int(np.linalg.norm(left - right)) # 두 좌표의 x좌표끼리의 차이와 y좌표끼리의 차이 중 긴놈을 반환해줌 (차이이므로 양수유지)

    # left
    mask_left_img = mask_img.crop((0, 0, width // 2, height))  # 0,0부터 width//2 , height 까지만 자르기
    mask_left_width = get_distance_from_point_to_line(left, nose, chin) # left라는 점과 nose, chin 두점을 지나는 교선 사이의 거리
    mask_left_width = int(mask_left_width * width_ratio)
    mask_left_img = mask_left_img.resize((mask_left_width, new_height))  # 왼쪽 마스크 resize

    # right
    mask_right_img = mask_img.crop((width // 2, 0, width, height))
    mask_right_width = get_distance_from_point_to_line(right, nose, chin)
    mask_right_width = int(mask_right_width * width_ratio)
    mask_right_img = mask_right_img.resize((mask_right_width, new_height))

    # merge mask
    size = (mask_left_img.width + mask_right_img.width, new_height) # 합칠 마스크 사이즈 추출
    mask_img = Image.new('RGBA', size)   # 새로운 이미지 파일 생성
    mask_img.paste(mask_left_img, (0, 0), mask_left_img)    # 이미지 파일에 왼쪽 마스크 붙이기
    mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img) # 이미지 파일에 오른쪽 마스크 붙이기

    # rotate mask
    angle = np.arctan2(chin[1] - nose[1], chin[0] - nose[0])  # arctan로 각 구한후
    rotated_mask_img = mask_img.rotate(angle, expand=True)  # 회전

    # calculate mask location
    center_x = (nose[0] + chin[0]) // 2
    center_y = (nose[1] + chin[1]) // 2

    offset = mask_img.width // 2 - mask_left_img.width
    radian = angle * np.pi / 180
    box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
    box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

    img2.paste(mask_img, (box_x, box_y), mask_img)

    file_name_path = 'test_data_final/bm/testbm' + str(count) + '.jpg' # 저장 경로및 파일명 설정
    img2.save(file_name_path)
    print(count)