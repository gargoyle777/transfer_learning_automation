import json
import albumentations as A
import cv2
import os
from pathlib import Path

parentDir = Path(__file__).parents[1] #Script -> AIPhoto
defaultSource = "resizedPhoto"
defaultDestination = "augmentedPhoto"
defaultIteration = 5

print("enter source directory (must be in AIPhoto):")
sourceDir = input()
if sourceDir != "":
    sourcePath = parentDir / sourceDir
    if not os.path.isdir(sourcePath):
        sourcePath = parentDir / defaultSource
else:
    sourcePath = parentDir / defaultSource

destinationPath = sourcePath / defaultDestination
os.mkdir(destinationPath)


jsonNewLabels = "{\"version\": 1,\"type\":\"bounding-box-labels\",\"boundingBoxes\":{"
readyList = []

transform = A.Compose(
    [
    A.CropAndPad(p=0.3 , percent=[-0.1 , 0.1], sample_independently=False),
    A.SafeRotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.SafeRotate(limit=20, p=0.15, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(p=0.35),
    A.Downscale(p=0.1),
    A.RandomSunFlare(p=0.1,src_radius=200),
    ], 
    bbox_params=A.BboxParams( format='coco', min_visibility=0.75 )
)
rePhotos = os.listdir(sourcePath)

labelsJson = json.loads( open(sourcePath / "bounding_boxes.labels", "r").read() ) 
photoCounter = 0

for rePhoto in rePhotos:    #unmodified photo
    iList = [] 
    if not rePhoto.endswith(".jpg"):
        continue
    else:
        photoCounter = photoCounter + 1
    imagePath =  str( sourcePath / rePhoto )
    image = cv2.imread( imagePath )    
    imgBoxes = []  
    labelList = labelsJson["boundingBoxes"][rePhoto]
    for singleLabel in labelList:        
        bbox = [singleLabel["label"], singleLabel["x"], singleLabel["y"], singleLabel["width"], singleLabel["height"] ]     
        imgBoxes.append(bbox)
    cv2.imwrite( str(destinationPath / rePhoto), image)     
    currentImage= "\"" + rePhoto + "\":["
    multiflag = 0
    for bbox in imgBoxes:
        if multiflag != 0 :
            currentImage = currentImage + ","
        else :
            multiflag = 1
        currentImage = currentImage + "{\"label\":\"" + bbox[0] + "\","
        currentImage = currentImage + "\"x\":" + str(int( bbox[1]) ) + "," + "\"y\":" + str( int(bbox[2]) ) + "," + "\"width\":" + str( int(bbox[3]) ) + "," + "\"height\":" + str( int(bbox[4]) ) +"}"
    currentImage = currentImage + "]"
    readyList.append(currentImage)

for i in range(defaultIteration):
    for rePhoto in rePhotos:
        iList = []     
        if not rePhoto.endswith(".jpg"):
            continue
        else:
            photoCounter = photoCounter + 1
        imageName =  str( sourcePath / rePhoto )
        image = cv2.imread( imageName )    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = []  
        labelList = labelsJson["boundingBoxes"][rePhoto]
        for singleLabel in labelList:        
            bbox = [singleLabel["x"], singleLabel["y"], singleLabel["width"], singleLabel["height"], singleLabel["label"] ]     
            bboxes.append(bbox)

        transformed = transform(image=image, bboxes=bboxes)
        transformed_image = transformed['image']
        t_bboxes = transformed['bboxes']
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)      #TODO :ho tolto la seconda linetta, dopo screen detection fixa
        splittedName = rePhoto.rsplit("-")
        newPhotoName = splittedName[0] +"-"+ str(photoCounter)+ "-" + ".jpg"
        cv2.imwrite( str(destinationPath / newPhotoName),transformed_image)     
        currentImage= "\"" + newPhotoName + "\":["
        multiflag = 0
        for t_bbox in t_bboxes:
            if multiflag != 0 :
                currentImage = currentImage + ","
            else :
                multiflag = 1
            currentImage = currentImage + "{\"label\":\"" + t_bbox[4] + "\","
            currentImage = currentImage + "\"x\":" + str(int( t_bbox[0]) ) + "," + "\"y\":" + str( int(t_bbox[1]) ) + "," + "\"width\":" + str( int(t_bbox[2]) ) + "," + "\"height\":" + str( int(t_bbox[3]) ) +"}"
        currentImage = currentImage + "]"
        readyList.append(currentImage)

#build the json without space (edge impulse bug)
multiflag = 0
for readybb in readyList :
    if multiflag != 0 :
        jsonNewLabels = jsonNewLabels + ","
    else:
        multiflag = 1
    jsonNewLabels = jsonNewLabels + readybb
jsonNewLabels = jsonNewLabels + "}}" # chiuso boundingBoxes  e fine json

# TODO what if file already exist?

file = open ( str( destinationPath / "bounding_boxes.labels" ) , "x")
file.write(jsonNewLabels)
file.close()