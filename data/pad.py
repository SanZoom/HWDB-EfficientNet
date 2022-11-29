import cv2
import numpy as np
import os

data_dir = '/home/oem/Desktop/yxk_workspace/Dataset/HWDB/train'
class_dir = [os.path.join(data_dir, c) for c in os.listdir(data_dir)]

count = 0
for c in class_dir:
    file_name = os.listdir(c)
    for fn in file_name:
        file_path = os.path.join(c, fn)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        if h > w:
            l = (h - w) // 2
            img = np.pad(img, ((0, 0),(l, l)), mode='constant', constant_values=255)
        else:
            l = (w - h) // 2
            img = np.pad(img, ((l, l), (0, 0)), mode='constant', constant_values=255)
        cv2.imwrite(file_path, img)
        count += 1

        if count % 1000 == 0:
            print(f'Have transformed {count}')