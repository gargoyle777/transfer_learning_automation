import os
from pathlib import Path
import shutil
parentDir = Path(__file__).parents[1] #Script -> AIPhoto

targetsPath = "C:\\Users\\nicco\\transfer_learning_automation\\imageClassification\\rawnegative"
allfiles=os.listdir(targetsPath)

for file in allfiles:
    if os.path.isdir(targetsPath+"\\"+file):
        continue
    srcpath=targetsPath+"\\"+file
    for i in range(9):
        dstpath=targetsPath+"\\copy"+str(i)+"_"+file
        shutil.copy(srcpath, dstpath)