from asyncio.windows_events import NULL
import os
import re
from turtle import st
import cv2
from xml.etree import ElementTree as et
import copy
from PIL import Image
import random

path = "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\intecsBraccio\\AIPhoto\\anExperiment"

files = os.listdir(path)

times = 50
namer = 0
for a in range(times):
    iconRequested = random.randint(3,56)
    for file in files:
        if file[-4:] != ".jpg":
            continue
        emptyIMG = []
        bboxes = []
        iconsIMG=[]
        img = Image.open(path+ "\\"+ file)
        xmlFile = file[0:-4]+".xml"
        ogtree = et.parse(path+"\\"+ xmlFile)
        ogroot = ogtree.getroot()
        ctree = copy.deepcopy(ogtree)
        croot = ctree.getroot()
        objects = ogroot.findall('object')
        for obj in objects:
            if obj.find("name").text == "emptyicon":
                xmin = int(obj.find('bndbox').find('xmin').text)
                ymin = int(obj.find('bndbox').find('ymin').text)
                xmax = int(obj.find('bndbox').find('xmax').text)
                ymax = int(obj.find('bndbox').find('ymax').text)
                bb = (xmin,ymin,xmax,ymax)
                emptyIMG.append(img.crop(bb))                
                bboxes.append(bb)
            if obj.find("name").text == "icon":
                xmin = int(obj.find('bndbox').find('xmin').text)
                ymin = int(obj.find('bndbox').find('ymin').text)
                xmax = int(obj.find('bndbox').find('xmax').text)
                ymax = int(obj.find('bndbox').find('ymax').text)
                bb = (xmin,ymin,xmax,ymax)
                iconsIMG.append(img.crop(bb))
                bboxes.append(bb)
        random.shuffle(bboxes)
        random.shuffle(iconsIMG)
        cimg = img.copy()
        newbboxes = []
        iconInserted = 0
        iconCounter = 0
        emptyCounter = 0
        for bbox in bboxes:
            if iconInserted == iconRequested:
                resEmpty = emptyIMG[emptyCounter].resize((bbox[2]-bbox[0],bbox[3]-bbox[1]))
                cimg.paste(resEmpty, box=bbox)
                emptyCounter+= 1
                if emptyCounter == len(emptyIMG):
                    emptyCounter = 0
                    random.shuffle(emptyIMG)
            else:
                resIcon = iconsIMG[iconCounter].resize((bbox[2]-bbox[0], bbox[3]-bbox[1]))
                cimg.paste(resIcon, box = bbox)
                iconCounter+=1
                if iconCounter == len(iconsIMG):
                    iconCounter = 0
                    random.shuffle(iconsIMG)
                newbboxes.append(bbox)
                iconInserted+=1
        croot.find('folder').text = "iconRand"
        croot.find('filename').text = "Rand-" +str(namer)+ croot.find('filename').text
        newPath = croot.find('path').text.split('\\')
        newPath[-1] = "iconRand\\Rand-"+ str(namer) + file
        croot.find('path').text = '\\'.join(newPath)
        objects = croot.findall('object')
        reinsertedCounter = 0
        for obj in objects:
            if obj.find('name').text == "screen":
                continue
            if obj.find('name').text == "innerScreen":
                continue
            if reinsertedCounter == iconInserted:
                croot.remove(obj)
                continue
            obj.find('name').text = "icon"
            obj.find('bndbox').find("xmin").text = str(newbboxes[reinsertedCounter][0]) 
            obj.find('bndbox').find("ymin").text = str(newbboxes[reinsertedCounter][1])
            obj.find('bndbox').find("xmax").text = str(newbboxes[reinsertedCounter][2])
            obj.find('bndbox').find("ymax").text = str(newbboxes[reinsertedCounter][3])
            reinsertedCounter+=1
        ctree.write(path + "\\iconRand\\Rand-"+ str(namer) + xmlFile)
        cimg.save(path+"\\iconRand\\Rand-"+str(namer) + file)
        namer+= 1