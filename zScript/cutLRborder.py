import os

import cv2 as cv2
import numpy as np

ogpath = "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\transferLearningCalc\\test_data"
paths =[ogpath]
dig = True      #dig in subdir

padColor = 255
padColor = [padColor]*3


for path in paths:
    files = os.listdir(path)
    for file in files:
        if os.path.isdir(path + "\\" + file) and dig:
            paths.append(path + "\\" + file)
            continue
        if file.endswith(".jpg"):
            img = cv2.imread(path + "\\" + file)
            (h,w) = img.shape[:2]
            scaled_img=img[0:1080, 420:1500]
            cv2.imwrite(path  + "\\" + file, scaled_img)