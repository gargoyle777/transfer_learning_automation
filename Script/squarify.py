import os

import cv2 as cv2
import numpy as np

ogpath = "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\imageClassification\\iconDataset"
paths =[ogpath]
dig = True      #dig in subdir

padColor = 255
padColor = [padColor]*3

th = 224 #target height i want to have
tw = 224 #target wideness i want to have
for path in paths:
    files = os.listdir(path)
    for file in files:
        if os.path.isdir(path + "\\" + file) and dig:
            paths.append(path + "\\" + file)
            continue
        if file.endswith(".jpg"):
            img = cv2.imread(path + "\\" + file)
            (h,w) = img.shape[:2]

            if h > th or w > tw: # shrinking image
                interp = cv2.INTER_AREA
            else: # stretching image
                interp = cv2.INTER_CUBIC

            aspect = w/h

            if aspect > 1: # horizontal
                new_w = tw
                new_h = np.round(new_w/aspect).astype(int)
                pad_vert = (th - new_h)/2
                pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
                pad_left, pad_right = 0, 0
            elif aspect < 1: # vertical image
                new_h = th
                new_w = np.round(new_h*aspect).astype(int)
                pad_horz = (tw-new_w)/2
                pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
                pad_top, pad_bot = 0, 0
            else: # square image
                new_h, new_w = th, tw
                pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

            # scale and pad
            scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
            scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE)
            cv2.imwrite(path  + "\\" + file, scaled_img)