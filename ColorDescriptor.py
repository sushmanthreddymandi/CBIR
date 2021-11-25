import numpy as np
import cv2
import imutils

class ColorDescriptor:
    def __init__(self, bins):
        self.bins = bins
        
    def describe(self, image, i=2):
        image = image.astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        (h, w) = image.shape[:2]                 
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        segements = [(0,cX,0,cY),(cX,w,0,cY),(cX,w,cY,h),(0,cX,cY,h)]
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        for(startX, endX, startY, endY) in segements:
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255 ,-1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)
            if i==1:
                cv2.imshow("Myimg",cornerMask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            hist = self.hist(image, cornerMask)
            features.extend(hist)
        if i==1:
            cv2.imshow("Myimg",ellipMask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        hist = self.hist(image, ellipMask)
        features.extend(hist)
        return features
    
    def hist(self,image,mask):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
        else:
            hist = cv2.normalize(hist, hist).flatten()
        return hist
