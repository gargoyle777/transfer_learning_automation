import albumentations as A
import cv2
import os

iteration = 3
counter = 0
sourcePath = "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\imageClassification\\iconDataset"
paths=[sourcePath]
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.8, brightness_limit=(-0.3,0.2)),
    A.SafeRotate(limit=5, p=0.2, border_mode=cv2.BORDER_REPLICATE),
    A.CropAndPad(percent = [-0.05, 0.05], p=0.5, pad_mode=cv2.BORDER_REPLICATE),
    A.Perspective(p=0.4, scale= 0.05,fit_output = True, pad_mode=cv2.BORDER_REPLICATE),
    A.FancyPCA(p=0.35),
    A.HorizontalFlip(p=0.35)
])

for path in paths:
    images = os.listdir(path)
    for imageName in images:
        if os.path.isdir(path + "\\" + imageName):
            if imageName == "valid":            #skippa valid
                continue
            else:
                paths.append(path + "\\" + imageName)
                continue
        if imageName[-4:] != ".jpg":
            print("skipping "+imageName)
            continue
        imagePath = path +"\\" +imageName 
        image = cv2.imread( imagePath )    
        image = cv2.resize(image,(264,264))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(iteration):
            counter = counter + 1
            transformed = transform(image = image )
            augmentedIMG = transformed["image"]
            imgName = imageName[0:-4] + "--" + str(counter) + ".jpg"
            augmentedIMG = cv2.cvtColor(augmentedIMG, cv2.COLOR_RGB2BGR)
            cv2.imwrite( path+"\\"+ imgName, augmentedIMG )
