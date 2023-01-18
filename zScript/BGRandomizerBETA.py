#randomized the background
from turtle import width
from PIL import Image
import os
from pathlib import Path
import re

parentDir = Path(__file__).parents[1] #Script -> AIPhoto
randDirPath = parentDir / "background"
targetsPath = "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\intecsBraccio\\AIPhoto\\transferLearning\\PhotoDetection\\train"
randDir = os.listdir(randDirPath)
targets = os.listdir(targetsPath)
destinationPath = targetsPath + "\\randomized"
os.mkdir(str(destinationPath))
targetList = []
targetLabels = []
for target in targets:
    if target.endswith(".jpg"):
        targetList.append(target)
        continue
    if target.endswith(".xml"):
        targetLabels.append(target)
imgCounter = 0
for image in targetList :
    boundingBoxes = []
    for label in targetLabels:
        if label[0:-4] != image[0:-4]:
            continue
        rawIMG = Image.open(str(targetsPath) + "\\" + image)
        xmlLabel = open(str(targetsPath) + "\\" + label, "r")
        while True:
            line = xmlLabel.readline()
            # check EOF
            if not line :
                break

            sRes = re.search("<width>.*</width>",line)
            if sRes != None :
                width = int(line.split(">")[1].split("<")[0])
                continue

            sRes = re.search("<height>.*</height>",line)
            if sRes != None :
                height = int(line.split(">")[1].split("<")[0])
                continue

            sRes = re.search("<xmin>.*</xmin>",line)
            if sRes != None :
                xmin = int(line.split(">")[1].split("<")[0])
                continue
            
            sRes = re.search("<ymin>.*</ymin>",line)
            if sRes != None :
                ymin = int(line.split(">")[1].split("<")[0])
                continue

            sRes = re.search("<xmax>.*</xmax>",line)
            if sRes != None :
                xmax = int(line.split(">")[1].split("<")[0])
                continue

            sRes = re.search("<ymax>.*</ymax>",line)
            if sRes != None :
                ymax = int(line.split(">")[1].split("<")[0])
                boundingBoxes.append([xmin,ymin,xmax,ymax])
    randomCounter = 0
    for randBG in randDir:
        rawRandom = Image.open(str(randDirPath) + "\\" + randBG)
        copyRandom = rawRandom.copy().resize((width, height))
        for bb in boundingBoxes:
            copyRandom.paste(rawIMG.crop((bb[0], bb[1], bb[2], bb[3] )), (bb[0], bb[1]))
        copyRandom.save(str(destinationPath) + "\\" + str(imgCounter) + "-" + str(randomCounter) + "-" + image)
        randomCounter = randomCounter + 1
    imgCounter = imgCounter + 1
