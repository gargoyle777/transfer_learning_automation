import os
import re
import pathlib
import shutil

path='C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\transferLearningCalc\\calcSRC'
files = os.listdir(path)
counter = 0
for file in files:
    if file[-4:] != ".jpg":
        continue
    xfile = file[0:-4] + ".xml"
    if counter % 5 == 0:
        shutil.copy2(path + '\\' + file, "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\transferLearningCalc\\PhotoDetection\\test\\"+file )
        shutil.copy2(path + '\\' + xfile, "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\transferLearningCalc\\PhotoDetection\\test\\"+xfile )
    else:
        shutil.copy2(path + '\\' + file, "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\transferLearningCalc\\PhotoDetection\\train\\"+file )
        shutil.copy2(path + '\\' + xfile, "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\transferLearningCalc\\PhotoDetection\\train\\"+xfile )
    counter += 1