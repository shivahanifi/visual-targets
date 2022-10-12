# Convert PPM to JPG
import os, sys
from PIL import Image
import glob

#list of ppm images
ListOfFiles = glob.glob('/Users/shiva.hnf/Documents/IIT/visual-targets/Demo/VT_DataAnnotation/VT_InputData/Images/TwoPeople_ThreeObjects/*.ppm')

#iterating over ppm images and converting them to jpg
for i in range(len(ListOfFiles)):
    infile = ListOfFiles[i]
    f, e = os.path.splitext(infile)
    outfile = f + ".jpg"
    if infile != outfile:
        try:
            with Image.open(infile) as im:
                im.save(outfile)
        except OSError:
            print("cannot convert", infile)
print(outfile)