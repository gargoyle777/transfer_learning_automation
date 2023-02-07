import os
import shutil

path='C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\transferLearningCalc\\calcSRC'   #src
testDir= "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\transferLearningCalc\\PhotoDetection\\test\\"
trainDir= "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\transferLearningCalc\\PhotoDetection\\train\\"
files = os.listdir(path)
counter = 0
for file in files:
    if file[-4:] != ".jpg":
        continue
    xfile = file[0:-4] + ".xml"
    if counter % 5 == 0:
        shutil.copy2(path + '\\' + file, testDir + file )
        shutil.copy2(path + '\\' + xfile, testDir + xfile )
    else:
        shutil.copy2(path + '\\' + file, trainDir + file )
        shutil.copy2(path + '\\' + xfile, trainDir + xfile )
    counter += 1