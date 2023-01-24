import os
import re
import pathlib
import shutil

paths=["C:\\Users\\nicco\\transfer_learning_automation\\imageClassification\\notcalc"]
dstpath= "C:\\Users\\nicco\\transfer_learning_automation\\imageClassification\\iconDataset"
counter = 0
scounter=0
if not os.path.exists(dstpath + "\\test"):
    os.makedirs(dstpath + "\\test")
if not os.path.exists(dstpath + "\\train"):
    os.makedirs(dstpath + "\\train")
if not os.path.exists(dstpath + "\\valid"):
    os.makedirs(dstpath + "\\valid")
for path in paths:
    images = os.listdir(path)
    for imageName in images:
        if os.path.isdir(path + "\\" + imageName):
            paths.append(path + "\\" + imageName)
            continue
        if imageName[-4:] != ".jpg":
            print("skipping "+imageName)
            continue
        if counter % 5 == 0:
            if scounter % 2 ==0:
                shutil.copy2(path + '\\' + imageName, dstpath+"\\test\\negative\\"+imageName )
            else:
                shutil.copy2(path + '\\' + imageName, dstpath+"\\valid\\negative\\"+imageName )
            scounter += 1
        else:
            shutil.copy2(path + '\\' + imageName, dstpath+"\\train\\negative\\"+imageName )
        counter += 1

    