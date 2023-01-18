import albumentations as A
import cv2
import os

iteration = 5
sourcePath = "C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\ScreenAI\\src\\cuttedBoxes"

destinationPath = sourcePath + "\\augmented"
os.mkdir(destinationPath)
transform = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.4),
    A.RandomBrightnessContrast(p=0.6),
    A.Downscale(p=0.2),
    A.SafeRotate(limit=15, p=0.2, border_mode=cv2.BORDER_REPLICATE),
])

images = os.listdir(sourcePath)

for imageName in images:
    if imageName[-4:] != ".jpg":
        print("skipping "+imageName)
        continue
    imagePath = sourcePath +"\\" +imageName 
    image = cv2.imread( imagePath )    
    image = cv2.resize(image,(264,264))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    counter = 0
    for i in range(iteration):
        counter = counter + 1
        transformed = transform(image = image )
        augmentedIMG = transformed["image"]
        imgName = imageName[0:-4] + "-" + str(counter) + ".jpg"
        augmentedIMG = cv2.cvtColor(augmentedIMG, cv2.COLOR_RGB2BGR)
        cv2.imwrite( destinationPath +"\\"+ imgName, augmentedIMG )
