from PIL import Image
import os
from pathlib import Path
import re
from xml.etree import ElementTree as et
import copy 

#could be adapted to any target

path =""
newdir = path + "\\screencut"
os.mkdir(newdir)
files = os.listdir(path)
for file in files:
    if file[-4:] == ".jpg":
        tree = et.parse(file[-4:] + ".xml")
        newTree = copy.deepcopy(tree)
        root = newTree.root()
        oldW = int(root.find('size').find("width").text)
        oldH = int(root.find('size').find("height").text)
        objects = root.findall('object')
        for member in objects:
            if member.find('name') != 'screen':
                continue
            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)
            img = Image.open(path +  '\\' + file)
            cropped = img.copy().crop((xmin,ymin,xmax,ymax))
            cropped.save(path + '\\screencut\\' + file[-4:]+"CROP"+".jpg")
            root.find('folder').text = "screencut"
            root.find('filename').text = file[-4:]+"CROP"+".jpg"
            root.find('path').text = path + '\\screencut\\' + file[-4:]+"CROP"+".jpg"
            root.find('size').find("width").text = str(xmax-xmin)
            root.find('size').find("height").text = str(ymax-ymin)
            for label in objects:
                if member.find('name') == 'screen':
                    continue
                member.find('bndbox').find('xmin').text = (int(member.find('bndbox').find('xmin').text) / oldW) * (xmax-xmin)
                member.find('bndbox').find('xmax').text = (int(member.find('bndbox').find('xmax').text) / oldW) * (xmax-xmin)
                member.find('bndbox').find('ymin').text = (int(member.find('bndbox').find('ymin').text) / oldH) * (ymax-ymin)
                member.find('bndbox').find('ymax').text = (int(member.find('bndbox').find('ymax').text) / oldH) * (ymax-ymin)
                newTree.write(path + '\\screencut\\' + file[-4:]+"CROP"+".xml")