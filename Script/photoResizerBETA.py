#cut and dowwnsize photo for EI
#fix their xml if available, but just enough to make the labeler work
from PIL import Image
import os
from pathlib import Path
import re

parentDir = Path(__file__).parents[1] #Script -> AIPhoto
defaultDestination = "resizedPhoto"

print("enter source directory (must be in AIPhoto):")
sourceDir = input()
sourcePath = parentDir / sourceDir
if not os.path.isdir(sourcePath):
    raise Exception("directory doesn't exist")

destinationPath = sourcePath / defaultDestination
os.mkdir(destinationPath)

rawFiles = os.listdir(sourcePath)

rawX = 1920
rawY =  1080

areaRES = (320,320)
resX = 320
resY = 320
# left, top, right, bottom
#areaFC = (500,80,1420,1000)        questo e' centrato
areaFC = (420,0,1500,1080)
for img in rawFiles :
    if img.endswith(".jpg"):
        rawIMG = Image.open(str(sourcePath) + "\\" + img)
        tmpIMG = rawIMG.crop(areaFC)
        tmpIMG = tmpIMG.resize(areaRES)
        tmpIMG.save(str(destinationPath) +"\\"+ img)

    if img.endswith(".xml"):
        newxml =""
        file = open(str(sourcePath) + "\\" + img, "r")
        while True:
            line = file.readline()
            # check EOF
            if not line :
                break
            addingLine = line
            sRes = re.search("<folder>.*</folder>",line)
            if sRes != None :
                addingLine = "<folder>" + str(destinationPath).rsplit("\\")[-1] + "</folder>"
                newxml = newxml + addingLine + "\n"
                continue

            sRes = re.search("<path>.*</path>",line)
            if sRes != None :
                addingLine = "<path>" + str(str(destinationPath) +"\\"+ img) + "</path>"
                newxml = newxml + addingLine + "\n"
                continue

            sRes = re.search("<width>.*</width>",line)
            if sRes != None :
                addingLine = "<width>" + "320" + "</width>"
                newxml = newxml + addingLine + "\n"
                continue

            sRes = re.search("<height>.*</height>",line)
            if sRes != None :
                addingLine = "<height>" + "320" + "</height>"
                newxml = newxml + addingLine + "\n"
                continue

            sRes = re.search("<xmin>.*</xmin>",line)
            if sRes != None :
                value = ((int(line[9:-8]) - areaFC[0]) / (areaFC[2] - areaFC[0])) * 320
                if value > 320:
                    value = 320
                if value < 0:
                    value = 0
                addingLine = "\t\t\t<xmin>" + str(int(value)) +"</xmin>"
                newxml = newxml + addingLine + "\n"
                continue

            sRes = re.search("<ymin>.*</ymin>",line)
            if sRes != None :
                value = ((int(line[9:-8]) - areaFC[1]) / (areaFC[3] - areaFC[1])) * 320
                if value > 320:
                    value = 320
                if value < 0:
                    value = 0
                addingLine = "\t\t\t<ymin>" + str(int(value)) +"</ymin>"
                newxml = newxml + addingLine + "\n"
                continue

            sRes = re.search("<xmax>.*</xmax>",line)
            if sRes != None :
                value = ((int(line[9:-8]) - areaFC[0]) / (areaFC[2] - areaFC[0])) * 320
                if value>320:
                    value = 320
                if value < 0:
                    value = 0
                addingLine = "\t\t\t<xmax>" + str(int(value)) +"</xmax>"
                newxml = newxml + addingLine + "\n"
                continue

            sRes = re.search("\t<ymax>.*</ymax>",line)
            if sRes != None :
                value = ((int(line[9:-8]) - areaFC[1]) / (areaFC[3] - areaFC[1])) * 320
                if value > 320:
                    value = 320
                if value < 0:
                    value = 0
                addingLine = "\t\t\t<ymax>" + str(int(value)) +"</ymax>"
                newxml = newxml + addingLine + "\n"
                continue

            newxml = newxml + addingLine
        filexml = open(str(destinationPath) +"\\"+ img, "w")
        filexml.write(newxml)
        filexml.close()
        