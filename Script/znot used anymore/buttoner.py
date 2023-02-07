import cv2
import os
from pathlib import Path
from xml.etree import ElementTree as et
import copy

path = "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\intecsBraccio\\AIPhoto\calculatorPhoto"
files = os.listdir(path)
for xml in files:
    if xml[-4:] != ".xml":
        continue
    tree = et.parse(path + "\\" + xml)
    newT = copy.deepcopy(tree)
    root = newT.getroot()
    objects = root.findall('object')
    for obj in objects:
        obj.find("name").text = "button"
    newT.write(path + "\\buttoned"+ xml)