import keras.backend as K
import tensorflow as tf
from keras.models import Model, Input
from icecream import ic
K.set_image_data_format('channels_first')
import csv
import os
from random import shuffle
from icecream import ic
import pdb
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import plot_model
import numpy as np
from tools.image_metrics import metrics_get
import matplotlib.pyplot as plt
import rasterio
import cv2
from skimage.transform import resize
class ImageReconstruction(object):

    def __init__ (self, tensor, output_c_dim, patch_size=256, overlap_percent=0):

        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.output_c_dim = output_c_dim
        self.model = model
    
    def Inference(self, tile):
        
        '''
        Normalize before call this method
        '''

        num_rows, num_cols, _ = tile.shape

        # Percent of overlap between consecutive patches.
        # The overlap will be multiple of 2
        overlap = round(self.patch_size * self.overlap_percent)
        overlap -= overlap % 2
        stride = self.patch_size - overlap
        
        # Add Padding to the image to match with the patch size and the overlap
        step_row = (stride - num_rows % stride) % stride
        step_col = (stride - num_cols % stride) % stride
 
        pad_tuple = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)), (0,0) )
        tile_pad = np.pad(tile, pad_tuple, mode = 'symmetric')

        # Number of patches: k1xk2
        k1, k2 = (num_rows+step_row)//stride, (num_cols+step_col)//stride
        print('Number of patches: %d x %d' %(k1, k2))

        # Inference
        probs = np.zeros((k1*stride, k2*stride, self.output_c_dim))

        for i in range(k1):
            for j in range(k2):
                
                patch = tile_pad[i*stride:(i*stride + self.patch_size), j*stride:(j*stride + self.patch_size), :]
                patch = patch[np.newaxis,...]
                # infer = self.sess.run(self.tensor, feed_dict={inputs: patch})

                infer = self.model.predict_on_batch(patch)

                probs[i*stride : i*stride+stride, 
                      j*stride : j*stride+stride, :] = infer[0, overlap//2 : overlap//2 + stride, 
                                                                overlap//2 : overlap//2 + stride, :]
            # print('row %d' %(i+1))

        # Taken off the padding
        probs = probs[:k1*stride-step_row, :k2*stride-step_col]

        return probs

class ImageLoading():
    def __init__(self, root_path = "D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/"):
        print("creating ImageLoading object...")
        self.root_path = root_path
        #imOptical = self.loadImage(root_path + "")
        print("Loading sar..")
        self.s1 = self.loadSar()
        print("Loading optical..")
        self.s2 = self.loadOptical()

        pdb.set_trace()

    def loadImage(self, path):
        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read()
        src.close()
        image[np.isnan(image)] = np.nanmean(image)
        ic(np.min(image), np.average(image), np.max(image))
        return image
    def loadOptical(self):
        s2_20m = self.loadImage(self.root_path + '2018_20m_b5678a1112.tif')
        s2_10m = self.loadImage(self.root_path + '2018_10m_b2348.tif')

        print(s2_20m.shape)

        s2_20m = np.transpose(s2_20m, (1, 2, 0))
        print(s2_20m.shape)

        dim = (s2_10m.shape[1], s2_10m.shape[2])
        #s2_20m = cv2.resize(s2_20m, dim, interpolation = cv2.INTER_AREA)
        s2_20m = resize(s2_20m, dim)
        s2_20m = np.transpose(s2_20m, (2, 0, 1))


        s2 = np.concatenate((s2_10m, s2_20m), axis=0)
        print(s2.shape)
        np.save('s2_2018_10m_20m.npy', s2)
        return s2

    def loadSar(self):

        #s1_vh_2018 = self.loadImage(self.root_path + 'cut_sent1_vh_2018.tif')
        #s1_vv_2018 = self.loadImage(self.root_path + 'cut_sent1_vv_2018.tif')
        s1_vh = self.loadImage(self.root_path + 'cut_sent1_vh_2019.tif')
        s1_vv = self.loadImage(self.root_path + 'cut_sent1_vv_2019.tif')

        s1 = np.concatenate((s1_vh, s1_vv), axis = 0)
        print(s1.shape)
        return s1
        
