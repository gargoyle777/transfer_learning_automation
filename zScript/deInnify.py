from asyncio.windows_events import NULL
import os
import re
from turtle import st
import cv2
from xml.etree import ElementTree as et
import copy
from PIL import Image
import random
from pathlib import Path

parentDir = Path(__file__).parents[0]
files = os.listdir(parentDir)
for file in files:
    if file.endswith(".xml"):
        tree=et.parse(parentDir / file)
        root = tree.getroot()
        objects = root.findall('object')
        for obj in objects:
            if obj.find("name").text=="innerScreen":
                root.remove(obj)
                tree.write(str(parentDir / file))
                print("removed from" + file)
