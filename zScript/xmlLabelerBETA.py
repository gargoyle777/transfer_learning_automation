from ensurepip import version
import os
import re
from pathlib import Path

parentDir = Path(__file__).parents[1] #Script -> AIPhoto
defaultSource = "resizedPhoto"

print("enter source directory (must be in AIPhoto):")
sourceDir = input()
if sourceDir != "":
    sourcePath = parentDir / sourceDir
    if not os.path.isdir(sourcePath):
        sourcePath = parentDir / defaultSource
else:
    sourcePath = parentDir / defaultSource 

xmlTargets = []
readyList = []
files = os.listdir(sourcePath)     #TODO resized se non sto testanto

for file in files:          # find .xml target
    if file.rsplit(".")[-1] == "xml":
        xmlTargets.append(file)

for target in xmlTargets:           # actual transformation

    file = open(sourcePath / target, "r")

    while True:
        line = file.readline()
        # check EOF
        if not line :
            break
        sRes = re.search("<filename>.*</filename>",line)
        if sRes != None :
            filename = sRes.group()[10:-11]
            break
    bboxes = []
    while True:
        line = file.readline()
        # check EOF
        if not line :
            break
        if line == "</object>":
            break
        sRes = re.search("<name>.*</name>",line)  
        if sRes != None :
            bboxname = sRes.group()[6:-7]
            continue
        sRes = re.search("<bndbox>",line)
        if sRes != None :     
            xmin = file.readline()[9:-8]    #conta i tab
            ymin = file.readline()[9:-8]
            xmax = file.readline()[9:-8]
            ymax = file.readline()[9:-8]
            bboxes.append("{\"label\":\"" + bboxname +"\",\"x\":" + xmin + "," + "\"y\":" + ymin + "," + "\"width\":" + str( int(xmax) - int(xmin) ) + "," + "\"height\":" + str( int(ymax) - int(ymin) ) +"}")
            continue
    readyList.append([filename, bboxes])
    file.close()

# build the json
jsonFile = "{\"version\": 1,\"type\": \"bounding-box-labels\",\"boundingBoxes\":{"
multiimages = 0
for images in readyList:
    if multiimages != 0:
            jsonFile = jsonFile + ","
    else:
        multiimages = 1
    jsonFile = jsonFile + "\"" + images[0] +"\":["
    multiboxes = 0
    for label in images[1]:
        if multiboxes != 0:
            jsonFile = jsonFile + ","
        else:
            multiboxes = 1
        jsonFile = jsonFile + label 
    jsonFile = jsonFile + "]"
jsonFile = jsonFile + "}}"
# TODO what if file already exist?
file = open (str (sourcePath / "bounding_boxes.labels"), "x")
file.write(jsonFile)
file.close()