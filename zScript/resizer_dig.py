import cv2
import os

paths=["C:\\Users\\nicco\\transfer_learning_automation\\imageClassification\\rawdata","C:\\Users\\nicco\\transfer_learning_automation\\imageClassification\\iconDataset"]

for path in paths:
    images = os.listdir(path)
    for imageName in images:
        if os.path.isdir(path + "\\" + imageName):
            paths.append(path + "\\" + imageName)
            continue
        if imageName[-4:] != ".jpg":
            print("skipping "+imageName)
            continue
        imagePath = path +"\\" +imageName 
        image = cv2.imread( imagePath )
        image=cv2.resize(image,(512,512))
        cv2.imwrite( path+"\\"+ imageName, image )