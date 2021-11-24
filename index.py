import ColorDescriptor
import argparse
import glob
import cv2
import numpy as np
from otherfeatures import getTextureFeatures
from skimage import io,img_as_ubyte
from skimage.color import colorconv

argParser  = argparse.ArgumentParser()
argParser .add_argument("-d","--dataset", required = True , help = "Path to directory that contains images")
argParser .add_argument("-i","--index", required = True , help = "Path to where the index will be stored")
args = vars(argParser .parse_args())
cd = ColorDescriptor.ColorDescriptor((8,12,13))
output = open(args["index"],"w")

for imgPath in glob.glob(args["dataset"]+"/*.jpg"):
    imageUID = imgPath[imgPath.rfind("/")+1:]
    image = cv2.imread(imgPath)
    features = cd.describe(image)
    img = io.imread(imgPath)
    features.extend(getTextureFeatures(img_as_ubyte(colorconv.rgb2gray(img))))
    features = [str(f) for f in features]
    output.write("%s,%s\n" % (imageUID, ",".join(features)))

output.close()
