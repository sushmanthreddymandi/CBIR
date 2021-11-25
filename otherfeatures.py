from skimage.feature.texture import greycomatrix, greycoprops
from math import pi
import cv2

def getTextureFeatures(image):
    features=greycoprops(greycomatrix(image, distances=[1, 2], angles=[0, pi/4, pi/2, 3*pi/4]), 'contrast')
    return cv2.normalize(features,features).flatten()