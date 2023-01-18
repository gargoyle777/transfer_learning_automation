#rename photo in target path, if i delete some of them, it names new photo in the holes

import os
from pathlib import Path

defaultSource = "screenDetection"

parentDir = Path(__file__).parents[1] #Script -> AIPhoto

sourcePath = "C:\\Users\\nicco\\Desktop\\singleicon\\cuttedBoxes"

print("enter new base name for photo:")
base = input()
if base == "":
    base = "PHOTO"
end = ".jpg"
counter = 0
toRename = []
fileList = os.listdir(sourcePath)
for element in fileList:
    if element.endswith(".jpg"):
        toRename.append(element)

for element in toRename:
    try:
        os.rename(sourcePath +"\\"+ element,  sourcePath + "\\" + base + "-" + str(counter) + end)
    except:
        toRename.append(element)
    counter = counter + 1
print("finished") 