import albumentations as A
import cv2
import os

iteration = 5
sourcePath = "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\imageClassification\\iconDataset\\valid"
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.8),
    A.Downscale(p=0.3),
    A.SafeRotate(limit=15, p=0.4, border_mode=cv2.BORDER_REPLICATE),
    A.CropAndPad(percent = [-0.05, 0.05], p=0.5, pad_mode=cv2.BORDER_REPLICATE),
    A.Perspective(p=0.4, scale= 0.05,fit_output = True, pad_mode=cv2.BORDER_REPLICATE)
])

dirs = os.listdir(sourcePath)
for dir in dirs:
    dirPath= sourcePath + "\\" + dir
    images= os.listdir(dirPath)
    for imageName in images:
        if imageName[-4:] != ".jpg":
            print("skipping "+imageName)
            continue
        if imageName[0:3]=="rec":
            print("skipping "+imageName)
            continue
        imagePath = dirPath +"\\" +imageName 
        image = cv2.imread( imagePath )    
        image = cv2.resize(image,(264,264))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        counter = 0
        for i in range(iteration):
            counter = counter + 1
            transformed = transform(image = image )
            augmentedIMG = transformed["image"]
            imgName = "rec"+imageName[0:-4] + "-" + str(counter) + ".jpg"
            augmentedIMG = cv2.cvtColor(augmentedIMG, cv2.COLOR_RGB2BGR)
            cv2.imwrite( dirPath +"\\"+ imgName, augmentedIMG )
