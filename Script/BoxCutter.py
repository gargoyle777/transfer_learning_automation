from PIL import Image
import os
from pathlib import Path
import re

defaultDestination = "cuttedBoxes"
sourcePath = "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\src\\freshimg"

if not os.path.isdir(sourcePath):
    raise Exception("directory doesn't exist")
destinationPath = sourcePath +"\\"+ defaultDestination
os.mkdir(destinationPath)

rawFiles = os.listdir(sourcePath)
images=[]
labels =[]
for file in rawFiles :
    if file.endswith(".jpg"):
        images.append(file)
    if file.endswith(".xml"):
        labels.append(file)
for image in images:
    for label in labels:
        if label[0:-4] != image[0:-4]:
            continue
        rawIMG = Image.open(str(sourcePath) + "\\" + image)
        xmlLabel = open(str(sourcePath) + "\\" + label, "r")
        while True:
            line = xmlLabel.readline()
            # check EOF
            if not line :
                break
            sRes = re.search("<name>.*</name>",line)
            if sRes != None :
                boxName = line.split(">")[1].split("<")[0]
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
                cuttedBox = rawIMG.crop((xmin,ymin,xmax,ymax))
                counter = 0
                cbNameBase = str(destinationPath) +"\\"+ "".join(boxName.split()) +"."+ image[0:-4] + "-"
                cbNameEnding = ".jpg"
                cbName = cbNameBase + str(counter) + cbNameEnding
                while Path(cbName).is_file():
                    counter = counter + 1
                    cbName = cbNameBase +  str(counter) + cbNameEnding
                cuttedBox.save(cbName)
                continue
    rawIMG.close()
    xmlLabel.close()