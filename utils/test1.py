import config as cfg
import os
import cv2
import numpy as np


for num in range(20, 40):
    image_path = os.path.join(cfg.DATA_PART1_PATH, 'image_1000')
    image_list = os.listdir(image_path)
    print(image_list[num])
    image = cv2.imread(os.path.join(image_path, image_list[num])).astype(np.uint8)
    # image = cv2.imread(os.path.join(image_path, 'TB1lHpnLXXXXXbwapXXunYpLFXX.jpg'))

    txt_path = os.path.join(cfg.DATA_PART1_PATH, 'txt_1000', image_list[num][:-3]+'txt')
    with open(txt_path) as f:
        lines = f.readlines()
        for line in lines:
            c = line.split(',')
            c = list(map(float, c[:-1]))
            c = list(map(int, c))
            print(c)
            pts = np.array(c)
            pts = pts.reshape(-1, 4, 2)
            print(pts)
            cv2.polylines(image, pts, True, (0, 0, 255), 2)


    cv2.imshow('test', image)
    cv2.waitKey()
