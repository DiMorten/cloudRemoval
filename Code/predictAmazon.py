import keras.backend as K
import tensorflow as tf
from keras.models import Model, Input
from icecream import ic
K.set_image_data_format('channels_first')
import csv
import os
from random import shuffle
import pdb
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import plot_model
import numpy as np
from tools.image_metrics import metrics_get
import matplotlib.pyplot as plt
import rasterio
import cv2
from skimage.transform import resize
from tools.image_metrics import metrics_get
class ImageReconstruction(object):

    def __init__ (self, model, output_c_dim, patch_size=256, overlap_percent=0):

        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.output_c_dim = output_c_dim
        self.model = model
    
    def infer(self, s2, s1):
        
        '''
        Normalize before call this method
        '''
        _, num_rows, num_cols = s2.shape
        _, num_rows, num_cols = s1.shape

        # Percent of overlap between consecutive patches.
        # The overlap will be multiple of 2
        overlap = round(self.patch_size * self.overlap_percent)
        overlap -= overlap % 2
        stride = self.patch_size - overlap
        
        # Add Padding to the image to match with the patch size and the overlap
        step_row = (stride - num_rows % stride) % stride
        step_col = (stride - num_cols % stride) % stride
 
        pad_tuple = ( (0,0), (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)) )
        s1_pad = np.pad(s1, pad_tuple, mode = 'symmetric')
        s2_pad = np.pad(s2, pad_tuple, mode = 'symmetric')

        # Number of patches: k1xk2
        k1, k2 = (num_rows+step_row)//stride, (num_cols+step_col)//stride
        print('Number of patches: %d x %d' %(k1, k2))

        # Inference
        probs = np.zeros((self.output_c_dim, k1*stride, k2*stride), dtype = np.float32)

        for i in range(k1):
            if i % 10 == 0:
                print("i = {}".format(i))
            for j in range(k2):
                
                patch_s1 = s1_pad[:, i*stride:(i*stride + self.patch_size), j*stride:(j*stride + self.patch_size)]
                patch_s2 = s2_pad[:, i*stride:(i*stride + self.patch_size), j*stride:(j*stride + self.patch_size)]
                
                patch_s1 = patch_s1[np.newaxis,...]
                patch_s2 = patch_s2[np.newaxis,...]
                
                # infer = self.sess.run(self.tensor, feed_dict={inputs: patch})

                predicted = self.model.predict_on_batch([patch_s2, patch_s1])[:, 0:13]

                probs[:, i*stride : i*stride+stride, 
                      j*stride : j*stride+stride] = predicted[0, :, overlap//2 : overlap//2 + stride, 
                                                                overlap//2 : overlap//2 + stride]
                # pdb.set_trace()
            # print('row %d' %(i+1))

        # Taken off the padding
        probs = probs[..., :k1*stride-step_row, :k2*stride-step_col]
        # probs = probs.astype(np.float32)
        np.save('probs.npy', probs)

        #pdb.set_trace()
        return probs

class Image():
    def __init__(self, root_path = "D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/10m_all_bands/"):
        print("creating ImageLoading object...")
        self.root_path = root_path
        #imOptical = self.loadImage(root_path + "")
        print("Loading sar..")
        self.s1 = self.loadSar()
        print("Loading optical..")

        path_list_s2 = ['Sentinel2_2018/COPERNICUS_S2_20180721_20180726_B1_B2_B3.tif',
            'Sentinel2_2018/COPERNICUS_S2_20180721_20180726_B4_B5_B6.tif',
            'Sentinel2_2018/COPERNICUS_S2_20180721_20180726_B7_B8_B8A.tif',
            'Sentinel2_2018/COPERNICUS_S2_20180721_20180726_B9_B10_B11.tif',
            'Sentinel2_2018/COPERNICUS_S2_20180721_20180726_B12.tif']
        print("Loading sentinel-2...")
        self.s2 = self.loadOptical(path_list_s2)

        path_list_s2_cloudy = ['Sentinel2_2018_Clouds/COPERNICUS_S2_20180611_B1_B2_B3.tif',
            'Sentinel2_2018_Clouds/COPERNICUS_S2_20180611_B4_B5_B6.tif',
            'Sentinel2_2018_Clouds/COPERNICUS_S2_20180611_B7_B8_B8A.tif',
            'Sentinel2_2018_Clouds/COPERNICUS_S2_20180611_B9_B10_B11.tif',
            'Sentinel2_2018_Clouds/COPERNICUS_S2_20180611_B12.tif']
        print("Loading sentinel-2 cloudy...")
        self.s2_cloudy = self.loadOptical(path_list_s2_cloudy)

        ic(np.min(self.s2[1]), np.average(self.s2[1]), np.max(self.s2[1]))
        ic(np.min(self.s2[2]), np.average(self.s2[2]), np.max(self.s2[2]))
        ic(np.min(self.s2[3]), np.average(self.s2[3]), np.max(self.s2[3]))
        
        ic(np.min(self.s2_cloudy[1]), np.average(self.s2_cloudy[1]), np.max(self.s2_cloudy[1]))
        ic(np.min(self.s2_cloudy[2]), np.average(self.s2_cloudy[2]), np.max(self.s2_cloudy[2]))
        ic(np.min(self.s2_cloudy[3]), np.average(self.s2_cloudy[3]), np.max(self.s2_cloudy[3]))
        
        #pdb.set_trace()

    def loadImage(self, path):
        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read()
        src.close()
        image[np.isnan(image)] = np.nanmean(image)
        #ic(np.min(image), np.average(image), np.max(image))
        return image
    def loadOptical(self, path_list):
        s2_b1_2_3 = self.loadImage(self.root_path + path_list[0])
        s2_b4_5_6 = self.loadImage(self.root_path + path_list[1])
        s2_b7_8_8A = self.loadImage(self.root_path + path_list[2])
        s2_b9_10_11 = self.loadImage(self.root_path + path_list[3])
        s2_b12 = self.loadImage(self.root_path + path_list[4])

        ic(s2_b1_2_3.shape, s2_b4_5_6.shape, s2_b7_8_8A.shape, s2_b9_10_11.shape, s2_b12.shape)
        print(s2_b1_2_3.shape, s2_b4_5_6.shape, s2_b7_8_8A.shape, s2_b9_10_11.shape, s2_b12.shape)

        #pdb.set_trace()
        s2 = np.concatenate((s2_b1_2_3, s2_b4_5_6, s2_b7_8_8A, s2_b9_10_11, s2_b12), axis=0)
        del s2_b1_2_3, s2_b4_5_6, s2_b7_8_8A, s2_b9_10_11, s2_b12
        print(s2.shape)

        s2[s2 < 0] = 0
        s2[s2 > 10000] = 10000
        #np.save('s2_2018.npy', s2)

        print(np.min(s2[1]), np.average(s2[1]), np.max(s2[1]))
        print(np.min(s2[2]), np.average(s2[2]), np.max(s2[2]))
        print(np.min(s2[3]), np.average(s2[3]), np.max(s2[3]))
        
        #pdb.set_trace()
        return s2

    def db2intensities(self, img):
        img = 10**(img/10.0)
        return img

    def loadSar(self):

        #s1_vh_2018 = self.loadImage(self.root_path + 'cut_sent1_vh_2018.tif')
        #s1_vv_2018 = self.loadImage(self.root_path + 'cut_sent1_vv_2018.tif')
        s1_vh = self.loadImage(self.root_path + 'COPERNICUS_S1_20180719_20180726_VH.tif')
        s1_vv = self.loadImage(self.root_path + 'COPERNICUS_S1_20180719_20180726_VV.tif')
        print(s1_vh.shape)

        s1_vv[s1_vv < -25] = -25
        s1_vv[s1_vv > 0] = 0
        s1_vh[s1_vh < -32.5] = -32.5
        s1_vh[s1_vh > 0] = 0

        s1 = np.concatenate((s1_vv, s1_vh), axis = 0)
        print(s1.shape)
        
        # checkimage for nans
        s1[np.isnan(s1)] = np.nanmean(s1)

        # s1 = self.db2intensities(s1)
        print(np.min(s1[0]), np.average(s1[0]), np.max(s1[0]))
        print(np.min(s1[1]), np.average(s1[1]), np.max(s1[1]))

        return s1
        
