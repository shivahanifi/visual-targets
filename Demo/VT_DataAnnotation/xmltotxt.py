# -*- coding: utf-8 -*-
"""XMLtoTXT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18cQkpEMZMWNovflh8aovxH05gFPYBdbi
"""

import xml.etree.ElementTree as ET
import glob

listOfFiles = glob.glob('/content/drive/MyDrive/One Person- Two Objects/*.xml')

f = open('bndbox_txt','w') #Creates a new file
#f.write('name, left, bottom, right, top\n')
f.close()

for i in range(len(listOfFiles)):
  tree = ET.parse(listOfFiles[i])
  print(tree)
  root = tree.getroot()
  print(root.tag, root.attrib)
  for ann in root.iter('annotation'):
    filename = ann.find('filename').text
    left = ann.find('object/bndbox/xmin').text
    right = ann.find('object').find('bndbox').find('xmax').text
    bottom = ann.find('object').find('bndbox').find('ymin').text
    top = ann.find('object').find('bndbox').find('ymax').text
    line_to_write = filename + ',' + left + ',' + bottom + ',' + right + ',' + top + '\n'
    with open('bndbox_txt', 'a') as f:
      f.write(line_to_write)