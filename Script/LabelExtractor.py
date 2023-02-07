import os
from unicodedata import name
from xml.etree import ElementTree as et

tree = et.parse("C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\intecsBraccio\\AIPhoto\\transferLearning\\PhotoDetection\\train\\Photo-0.xml") #xml path
root = tree.getroot()
print("\'background\'",end="")
counter = 1
for member in root.findall('object'):
    counter+=1
    print(f",\'{member.find('name').text}\'", end ="")
print("\n" + str(counter))