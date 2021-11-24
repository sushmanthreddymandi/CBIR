import ColorDescriptor
import argparse
import cv2
import numpy as np
import csv
from otherfeatures import getTextureFeatures
from skimage import io,img_as_ubyte
from skimage.color import colorconv

class Searcher:
    def __init__(self,indexPath):
        self.indexPath = indexPath
        
    def search(self,queryFeatures):
        results = {}
        with open(self.indexPath) as f:
            reader = csv.reader(f)
            for row in reader:
                features = [float(x) for x in row[1:]]
                d = self.chiDistanceCal(features, queryFeatures)
                results[row[0]] = d
            f.close()
        results = sorted([(v,k) for (k,v) in results.items()])
        return results[:1]
    
    def chiDistanceCal(self, histA, histB, eps = 1e-10):
        d = 0.5 * np.sum([((a-b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
        return d



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True, help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True, help = "Path to the result path")
args = vars(ap.parse_args())
cd = ColorDescriptor.ColorDescriptor((8,12,13))
query = cv2.imread(args["query"])
queryFeatures = cd.describe(query)
img = io.imread(imgPath);
queryFeatures.extend(getTextureFeatures(img_as_ubyte(colorconv.rgb2gray(img))))
s1 = Searcher(args["index"])
results = s1.search(queryFeatures)
cv2.imshow("Query",query)


for (score, resultID) in results:
    result1 = cv2.imread(args["result_path"] + "/" + resultID)
    result = cv2.resize(result1,(300,300))
    cv2.imshow("Result",result)
    cv2.waitKey(0)
