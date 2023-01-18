from fileinput import filename
import albumentations as A
import cv2
import os
from pathlib import Path
from xml.etree import ElementTree as et
import copy

photoPath = "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\transferLearningCalc\\calcSRC"
augmentIteration = 4
#A.CropAndPad(p=0.3 , percent=[-0.1 , 0.1], sample_independently=False),
#   A.SafeRotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
#    A.SafeRotate(limit=20, p=0.15, border_mode=cv2.BORDER_CONSTANT),
#    A.RandomBrightnessContrast(p=0.35),
#    A.Downscale(p=0.1),
#    A.RandomSunFlare(p=0.1,src_radius=200),
transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.8, brightness_limit=(-0.3,0.2)),
        A.SafeRotate(limit=5, p=0.2, border_mode=cv2.BORDER_REPLICATE),
        A.CropAndPad(percent = [-0.05, 0.05], p=0.5, pad_mode=cv2.BORDER_REPLICATE),
        A.Perspective(p=0.4, scale= 0.05,fit_output = True, pad_mode=cv2.BORDER_REPLICATE),
        A.FancyPCA(p=0.5),
        A.HorizontalFlip(p=0.35)
    ], 
    bbox_params=A.BboxParams(format= 'pascal_voc')
)
files = os.listdir(photoPath)
photos = []
allLabels = []
for file in files:
    if file[-4:] ==".jpg":
        labels=[]
        photos.append(file)
        tree = et.parse(photoPath + "\\" + (file[0:-4] + ".xml"))
        root = tree.getroot()
        objects = root.findall('object')
        for member in objects:
            name = member.find("name").text
            # xmin = left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # xmax = right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # ymin = left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # ymax = right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)
            bb = [xmin,ymin,xmax,ymax,name]
            labels.append(bb)
        allLabels.append(labels)
counter = 0
for i in range(augmentIteration):
    photoCounter=0
    for photo in photos:
        image = cv2.imread(photoPath +"\\" + photo)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, bboxes = allLabels[photoCounter])
        transformed_image = transformed['image']
        t_bboxes = transformed['bboxes']
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(photoPath+"\\"+str(counter)+photo,transformed_image)
        newXml = copy.deepcopy(et.parse(photoPath +"\\"+ (photo[0:-4] + ".xml")))
        newRoot = newXml.getroot()
        newRoot.find('filename').text= str(counter)+photo
        bbcounter=0
        for member in newRoot.findall('object'):
            member.find('name').text = t_bboxes[bbcounter][4]
            member.find('bndbox').find('xmin').text = str(int(t_bboxes[bbcounter][0]))
            member.find('bndbox').find('ymin').text = str(int(t_bboxes[bbcounter][1]))
            member.find('bndbox').find('xmax').text = str(int(t_bboxes[bbcounter][2]))
            member.find('bndbox').find('ymax').text = str(int(t_bboxes[bbcounter][3]))
            bbcounter+=1
        newXml.write(photoPath + "\\" + str(counter) + photo[:-4] + ".xml")
        photoCounter+=1
        counter+=1