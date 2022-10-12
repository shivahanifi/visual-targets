#Convert XML to TXT 
import xml.etree.ElementTree as ET
import glob

#list of xml files
listOfFiles = sorted(glob.glob('/Users/shiva.hnf/Documents/IIT/visual-targets/Demo/VT_DataAnnotation/VT_InputData/Annotations/TwoPeople_ThreeObjects(2)/*.xml'))

#opening a txt file
f = open('/Users/shiva.hnf/Documents/IIT/visual-targets/Demo/VT_DataAnnotation/bndBox_TXT/TwoPeople_ThreeObjects(2).txt','w') #Creates a new file
#f.write('name, left, bottom, right, top\n')
f.close()

#parsing the xml files
for i in range(len(listOfFiles)):
  tree = ET.parse(listOfFiles[i])
  root = tree.getroot()
  #extracting info from xml files
  for ann in root.iter('annotation'):
    filename = ann.find('filename').text
    left = ann.find('object/bndbox/xmin').text
    right = ann.find('object/bndbox/xmax').text
    bottom = ann.find('object/bndbox/ymin').text
    top = ann.find('object/bndbox/ymax').text
    line_to_write = filename + ',' + left + ',' + bottom + ',' + right + ',' + top + '\n'
    #writing to the txt file
    with open('/Users/shiva.hnf/Documents/IIT/visual-targets/Demo/VT_DataAnnotation/bndBox_TXT/TwoPeople_ThreeObjects(2).txt', 'a') as f:
      f.write(line_to_write) 