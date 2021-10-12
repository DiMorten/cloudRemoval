
import argparse
import random
from icecream import ic
import pdb
import pickle
ic.configureOutput(includeContext=True)
import rasterio
import numpy as np

root_path="D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/ROIs1868_summer_s1/s1_86/ROIs1868_summer_s1_86_p58.tif"
#root_path
def loadImage(path):
    src = rasterio.open(path, 'r', driver='GTiff')
    image = src.read()
    src.close()
    image[np.isnan(image)] = np.nanmean(image)
    #ic(np.min(image), np.average(image), np.max(image))
    return image

im = loadImage(root_path)
print(im.shape)
print(np.min(im[0]), np.average(im[0]), np.max(im[0]))
print(np.min(im[1]), np.average(im[1]), np.max(im[1]))
print(np.min(im), np.average(im), np.max(im))