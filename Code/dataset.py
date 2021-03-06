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
import matplotlib.pyplot as plt 
import tifffile as tiff
import pathlib
import gdal

def loadImage(path):
    # Read tiff Image
    ic(path)
    gdal_header = gdal.Open(path)
    image = gdal_header.ReadAsArray()
    return image

def make_dir(dir_path):
    if os.path.isdir(dir_path):
        print("WARNING: Folder {} exists and content may be overwritten!")
    else:
        os.makedirs(dir_path)

    return dir_path


class ImageReconstruction(object):

    def __init__ (self, model, output_c_dim, patch_size=256, overlap_percent=0,
        loadIms = False):

        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.output_c_dim = output_c_dim
        self.model = model
        self.loadIms = loadIms
    
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
            if i % 5 == 0:
                print("i = {}".format(i))
            for j in range(k2):
                
                patch_s1 = s1_pad[:, i*stride:(i*stride + self.patch_size), j*stride:(j*stride + self.patch_size)]
                patch_s2 = s2_pad[:, i*stride:(i*stride + self.patch_size), j*stride:(j*stride + self.patch_size)]
                
                patch_s1 = patch_s1[np.newaxis,...]
                patch_s2 = patch_s2[np.newaxis,...]
                
                # infer = self.sess.run(self.tensor, feed_dict={inputs: patch})
                if patch_s2.shape[-1] != 1024:
                    ic(patch_s1.shape, patch_s2.shape, i, j, i*stride, i*stride + self.patch_size,
                        j*stride, j*stride + self.patch_size)
                    pdb.set_trace()
                predicted = self.model.predict_on_batch([patch_s2, patch_s1])[:, 0:self.output_c_dim]
                probs[:, i*stride : i*stride+stride, 
                      j*stride : j*stride+stride] = predicted[0, :, overlap//2 : overlap//2 + stride, 
                                                                overlap//2 : overlap//2 + stride]
                # pdb.set_trace()
            # print('row %d' %(i+1))

        # Taken off the padding
        probs = probs[..., :k1*stride-step_row, :k2*stride-step_col]
        # probs = probs.astype(np.float32)
        #np.save('probs.npy', probs)

        #pdb.set_trace()
        return probs

class Image():
    def __init__(self,
        date = '2018', crop_sample_im = False, normalize = True, loadIms = False):
        #self.site = site # PA,MG
        self.loadIms = loadIms

        self.normalize = normalize
        self.crop_sample_im = crop_sample_im
        # self.root_path = root_path

        print("creating ImageLoading object...")


        scale = 2000
        max_val_sar = 2
        clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        clip_max = [[0, 0], [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                    [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]

        self.max_val = max_val_sar
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.scale = scale
        self.date = date
        #imOptical = self.loadImage(root_path + "")
        #self.loadIms=False
        if self.loadIms == False:
            print("Loading sar..")
            self.s1 = self.loadSar()
            print("Loading optical..")
            np.save('s1_'+self.date+'_'+self.site+'.npy', self.s1)


            print("Loading sentinel-2...")
            self.s2 = self.loadOptical(self.path_list_s2, 'S2/')
            np.save('s2_'+self.date+'_'+self.site+'.npy', self.s2)

            print("Loading sentinel-2 cloudy...")
            self.s2_cloudy = self.loadOptical(self.path_list_s2_cloudy, 'S2_cloudy/')
            np.save('s2_cloudy_'+self.date+'_'+self.site+'.npy', self.s2_cloudy)
        else:
            self.s1 = np.load('s1_'+self.date+'_'+self.site+'.npy')
            self.s2 = np.load('s2_'+self.date+'_'+self.site+'.npy')
            self.s2_cloudy = np.load('s2_cloudy_'+self.date+'_'+self.site+'.npy')
        
        if self.site == 'MG':
            self.s1 = self.s1[:, :-4000, 3000:]
            ic(self.s1.shape)
        #self.s1 = self.s1[:,1000:1000+512, 1000:1000+512]
        #self.s2 = self.s2[:,1000:1000+512, 1000:1000+512]
        #self.s2_cloudy = self.s2_cloudy[:,1000:1000+512, 1000:1000+512]

        self.crop0 = 1000
        #self.crop0 = 3000
        #self.delta_crop = 1500
        self.delta_crop = 1500
        
        if self.crop_sample_im == True:
            self.s1 = self.s1[:,self.crop0:self.crop0+self.delta_crop, self.crop0:self.crop0+self.delta_crop]
            self.s2 = self.s2[:,self.crop0:self.crop0+self.delta_crop, self.crop0:self.crop0+self.delta_crop]
            self.s2_cloudy = self.s2_cloudy[:,self.crop0:self.crop0+self.delta_crop, self.crop0:self.crop0+self.delta_crop]

        _, self.rows, self.cols = self.s1.shape
        # ic(np.min(self.s2), np.average(self.s2), np.std(self.s2), np.max(self.s2))

        # ic(np.min(self.s1[1]), np.average(self.s1[1]), np.max(self.s1[1]))

        
        ##s2_copy = np.transpose(self.s2, (1, 2, 0))
        ##ic(s2_copy[:,:,1:4].shape)
        ##tiff.imsave('s2_saved_first.tif', s2_copy[:,:,1:4].astype(np.int16), photometric='rgb')
        
        ##s2_cloudy_copy = np.transpose(self.s2_cloudy, (1, 2, 0))
        ##ic(s2_cloudy_copy[:,:,1:4].shape)
        ##tiff.imsave('s2_cloudy_saved_first.tif', s2_cloudy_copy[:,:,1:4].astype(np.int16), photometric='rgb')
        
        ##plt.figure(figsize=(5,10))
        ##plt.imshow(s2_copy[:,:,1:4].astype(np.int16))
        ##plt.axis('off')
        ##plt.savefig('s2_rgb.png')
        #plt.show()
        #pdb.set_trace()     
        '''
        ic(np.min(self.s2[1]), np.average(self.s2[1]), np.max(self.s2[1]))
        ic(np.min(self.s2[2]), np.average(self.s2[2]), np.max(self.s2[2]))
        ic(np.min(self.s2[3]), np.average(self.s2[3]), np.max(self.s2[3]))
        
        ic(np.min(self.s2_cloudy[1]), np.average(self.s2_cloudy[1]), np.max(self.s2_cloudy[1]))
        ic(np.min(self.s2_cloudy[2]), np.average(self.s2_cloudy[2]), np.max(self.s2_cloudy[2]))
        ic(np.min(self.s2_cloudy[3]), np.average(self.s2_cloudy[3]), np.max(self.s2_cloudy[3]))
        '''
        # self.s1_unnormalized = self.s1.copy()
        if self.normalize == True:
            self.s1 = self.get_normalized_data(self.s1, data_type = 1)
            self.s2_cloudy = self.get_normalized_data(self.s2_cloudy, data_type = 2)
            self.s2 = self.get_normalized_data(self.s2, data_type = 2)

        ic(np.min(self.s1[1]), np.average(self.s1[1]), np.max(self.s1[1]))
        ic(np.min(self.s2), np.average(self.s2), np.max(self.s2))

        #pdb.set_trace()

    def loadImage(self, path):
        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read()
        src.close()
        image[np.isnan(image)] = np.nanmean(image)
        #ic(np.min(image), np.average(image), np.max(image))
        return image

#        def loadOptical()
    def db2intensities(self, img):
        img = 10**(img/10.0)
        return img

    def loadSar(self):

        #s1_vh_2018 = self.loadImage(self.root_path + 'cut_sent1_vh_2018.tif')
        #s1_vv_2018 = self.loadImage(self.root_path + 'cut_sent1_vv_2018.tif')
        
        s1_vh = self.loadImage(self.root_path + self.path_list_s1[0])
        s1_vv = self.loadImage(self.root_path + self.path_list_s1[1])

        print(s1_vh.shape)

        s1_vv[s1_vv < -25] = -25
        s1_vv[s1_vv > 0] = 0
        s1_vh[s1_vh < -32.5] = -32.5
        s1_vh[s1_vh > 0] = 0

        s1 = np.concatenate((s1_vv, s1_vh), axis = 0)
        ic(s1.shape)
        #pdb.set_trace()
        
        # checkimage for nans
        s1[np.isnan(s1)] = np.nanmean(s1)

        # s1 = self.db2intensities(s1)
        print(np.min(s1[0]), np.average(s1[0]), np.max(s1[0]))
        print(np.min(s1[1]), np.average(s1[1]), np.max(s1[1]))

        return s1

    def get_normalized_data(self, data_image, data_type):

        # SAR
        if data_type == 1:
            ic(np.min(data_image[0]), np.average(data_image[0]), np.std(data_image[0]), np.max(data_image[0]))
            ic(data_image.shape)
#            pdb.set_trace()
            for channel in range(len(data_image)):
                ic(data_image.shape, self.clip_min, self.clip_max, data_type, channel)
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (
                        self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
            ic(np.min(data_image[0]), np.average(data_image[0]), np.std(data_image[0]), np.max(data_image[0]))

        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])

            data_image /= self.scale

        return data_image
    

    #generate_output_images(data_image, ID[i], predicted_images_path, input_data_folder, cloud_threshold)
    def saveSampleIms(self, s1, s2, s2_cloudy, predicted):
        # predicted *= self.scale

        self.generate_output_images(s1, s2, s2_cloudy, predicted * self.scale)


    def generate_output_images(self, s1, s2, s2_cloudy, predicted, 
        predicted_images_path = 'sample_ims', scene_name = '2018'):

        sar_preview = self.get_preview(s1, True, [1, 2, 2], sar_composite=True)

        opt_bands = [4, 3, 2]  # R, G, B bands (S2 channel numbers)

        cloudFree_preview = self.get_preview(s2, True, opt_bands, brighten_limit=2000)
        cloudy_preview = self.get_preview(s2_cloudy, True, opt_bands)
        
        predicted_preview = self.get_preview(predicted, True, opt_bands, 2000)

        ic(np.min(predicted_preview), np.average(predicted_preview), np.max(predicted_preview))

        #predicted_images_path = 'sample_ims'
        #scene_name = '2018'
        out_path = make_dir(os.path.join(predicted_images_path, scene_name))

        self.save_single_images(sar_preview, cloudy_preview, cloudFree_preview, 
            predicted_preview, out_path)

        return


    def get_preview(self, file, predicted_file, bands, brighten_limit=None, sar_composite=False):
        if not predicted_file:
            with rasterio.open(file) as src:
                r, g, b = src.read(bands)
        else:
            # file is actually the predicted array
            r = file[bands[0] - 1]
            g = file[bands[1] - 1]
            b = file[bands[2] - 1]
        # ic(r.shape, g.shape, b.shape)
        if brighten_limit is None:
            return self.get_rgb_preview(r, g, b, sar_composite)
        else:
            r = np.clip(r, 0, brighten_limit)
            g = np.clip(g, 0, brighten_limit)
            b = np.clip(b, 0, brighten_limit)
            return self.get_rgb_preview(r, g, b, sar_composite)


    def save_single_images(self, sar_preview, cloudy_preview, cloudFree_preview, 
            predicted_preview, out_path):

        pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
        ##ic(predicted_preview.shape, cloudFree_preview.shape)
    #    pdb.set_trace()
        self.save_single_image(sar_preview, out_path, "inputsar")
        self.save_single_image(cloudy_preview, out_path, "input")
        self.save_single_image(cloudFree_preview, out_path, "inputtarg")
        self.save_single_image(predicted_preview, out_path, "inputpred")

        return

    def save_single_image(self, image, out_path, name):
        plt.figure(frameon=False)
        plt.imshow(image)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.axis('off')
        plt.savefig(os.path.join(out_path, name + '.png'), dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()

        return

    def get_rgb_preview(self, r, g, b, sar_composite=False):
        if not sar_composite:

            # stack and move to zero
            rgb = np.dstack((r, g, b))
            rgb = rgb - np.nanmin(rgb)

            # treat saturated images, scale values
            if np.nanmax(rgb) == 0:
                rgb = 255 * np.ones_like(rgb)
            else:
                rgb = 255 * (rgb / np.nanmax(rgb))

            # replace nan values before final conversion
            rgb[np.isnan(rgb)] = np.nanmean(rgb)

            return rgb.astype(np.uint8)

        else:
            # generate SAR composite
            HH = r
            HV = g
            ic(np.min(HH), np.average(HH), np.std(HH),np.max(HH))

            HH = np.clip(HH, -25.0, 0)
            HH = (HH + 25.1) * 255 / 25.1
            HV = np.clip(HV, -32.5, 0)
            HV = (HV + 32.6) * 255 / 32.6
            ic(np.min(HH), np.average(HH), np.std(HH),np.max(HH))

            rgb = np.dstack((np.zeros_like(HH), HH, HV))

            return rgb.astype(np.uint8)

    def loadMask(self):
        self.mask = np.load(self.root_path + self.mask_filename)
        if self.crop_sample_im == True:
            self.mask = self.mask[self.crop0:self.crop0+self.delta_crop, self.crop0:self.crop0+self.delta_crop]

    def addPadding(self, patch_size, overlap_percent):

        #if self.padding == True:

        # Percent of overlap between consecutive patches.

        overlap = round(patch_size * overlap_percent)
        overlap -= overlap % 2
        stride = patch_size - overlap
        # Add Padding to the image to match with the patch size
        step_row = (stride - self.rows % stride) % stride
        step_col = (stride - self.cols % stride) % stride
        pad_tuple_msk = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)) )
        pad_tuple_im = ( (0,0), (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)) )
        
        self.s1 = np.pad(self.s1, pad_tuple_im, mode = 'symmetric')
        self.s2 = np.pad(self.s2, pad_tuple_im, mode = 'symmetric')
        self.s2_cloudy = np.pad(self.s2_cloudy, pad_tuple_im, mode = 'symmetric')
        try:
            self.mask = np.pad(self.mask, pad_tuple_msk, mode = 'symmetric')
        except:
            print("No mask")

        if self.site == 'PA':
            self.root_path = "D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/10m_all_bands/"
        elif self.site == 'MG':
            self.root_path = "D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/10m_all_bands_MG/"


class ImagePA(Image):
    def __init__(self, date, crop_sample_im, normalize, loadIms, 
            root_path = "D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/10m_all_bands/"):
        self.root_path = root_path
        self.site = 'PA'
        self.mask_filename = 'tile_mask_0tr_1vl_2ts.npy'

        if date == '2018':
            self.path_list_s2 = ['Sentinel2_2018/COPERNICUS_S2_20180721_20180726_B1_B2_B3.tif',
                'Sentinel2_2018/COPERNICUS_S2_20180721_20180726_B4_B5_B6.tif',
                'Sentinel2_2018/COPERNICUS_S2_20180721_20180726_B7_B8_B8A.tif',
                'Sentinel2_2018/COPERNICUS_S2_20180721_20180726_B9_B10_B11.tif',
                'Sentinel2_2018/COPERNICUS_S2_20180721_20180726_B12.tif']
            self.path_list_s2_cloudy = ['Sentinel2_2018_Clouds/COPERNICUS_S2_20180611_B1_B2_B3.tif',
                'Sentinel2_2018_Clouds/COPERNICUS_S2_20180611_B4_B5_B6.tif',
                'Sentinel2_2018_Clouds/COPERNICUS_S2_20180611_B7_B8_B8A.tif',
                'Sentinel2_2018_Clouds/COPERNICUS_S2_20180611_B9_B10_B11.tif',
                'Sentinel2_2018_Clouds/COPERNICUS_S2_20180611_B12.tif']
            self.path_list_s1 = ['COPERNICUS_S1_20180719_20180726_VH.tif',
                'COPERNICUS_S1_20180719_20180726_VV.tif']
        else:
            self.path_list_s2 = ['Sentinel2_2019/COPERNICUS_S2_20190721_20190726_B1_B2_B3.tif',
                'Sentinel2_2019/COPERNICUS_S2_20190721_20190726_B4_B5_B6.tif',
                'Sentinel2_2019/COPERNICUS_S2_20190721_20190726_B7_B8_B8A.tif',
                'Sentinel2_2019/COPERNICUS_S2_20190721_20190726_B9_B10_B11.tif',
                'Sentinel2_2019/COPERNICUS_S2_20190721_20190726_B12.tif']
            self.path_list_s2_cloudy = ['Sentinel2_2019_Clouds/COPERNICUS_S2_20190706_B1_B2_B3.tif',
                'Sentinel2_2019_Clouds/COPERNICUS_S2_20190706_B4_B5_B6.tif',
                'Sentinel2_2019_Clouds/COPERNICUS_S2_20190706_B7_B8_B8A.tif',
                'Sentinel2_2019_Clouds/COPERNICUS_S2_20190706_B9_B10_B11.tif',
                'Sentinel2_2019_Clouds/COPERNICUS_S2_20190706_B12.tif']   
            self.path_list_s1 = ['COPERNICUS_S1_20190721_20190726_VH.tif',
                'COPERNICUS_S1_20190721_20190726_VV.tif']
             
        super().__init__(date, crop_sample_im, normalize, loadIms)
    def loadOptical(self, path_list, folder):
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

class ImageMG(Image):
    def __init__(self, date, crop_sample_im, normalize, loadIms, 
            root_path = "D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/10m_all_bands_MG/"):
        self.site = 'MG'
        self.root_path = root_path
        # self.mask_filename = 'ref_2019_2020_20798x13420.npy'
        self.mask_filename = 'MT_tr_0_val_1_ts_2_16795x10420_new.npy'
        if date == '2019':
            self.path_list_s2 = ['Sentinel2_2019/S2_R1_MT_2019_08_02_2019_08_05_B1_B2-008.tif',
                'Sentinel2_2019/S2_R1_MT_2019_08_02_2019_08_05_B3_B4-009.tif',
                'Sentinel2_2019/S2_R1_MT_2019_08_02_2019_08_05_B5_B6-013.tif',
                'Sentinel2_2019/S2_R1_MT_2019_08_02_2019_08_05_B7_B8-014.tif',
                'Sentinel2_2019/S2_R1_MT_2019_08_02_2019_08_05_B8A_B9-003.tif',
                'Sentinel2_2019/S2_R1_MT_2019_08_02_2019_08_05_B10_B11-006.tif',
                'Sentinel2_2019/S2_R1_MT_2019_08_02_2019_08_05_B12.tif']
            self.path_list_s2_cloudy = ['Sentinel2_2019_Clouds/S2CL_R1_MT_2019_09_26_2019_09_29_B1_B2-002.tif',
                'Sentinel2_2019_Clouds/S2CL_R1_MT_2019_09_26_2019_09_29_B3_B4-007.tif',
                'Sentinel2_2019_Clouds/S2CL_R1_MT_2019_09_26_2019_09_29_B5_B6-003.tif',
                'Sentinel2_2019_Clouds/S2CL_R1_MT_2019_09_26_2019_09_29_B7_B8-006.tif',
                'Sentinel2_2019_Clouds/S2CL_R1_MT_2019_09_26_2019_09_29_B8A_B9-004.tif',
                'Sentinel2_2019_Clouds/S2CL_R1_MT_2019_09_26_2019_09_29_B10_B11-005.tif',
                'Sentinel2_2019_Clouds/S2CL_R1_MT_2019_09_26_2019_09_29_B12.tif']
            
            self.path_list_s1 = ['S1_R1_MT_2019_08_02_2019_08_09_VH.tif',
                'S1_R1_MT_2019_08_02_2019_08_09_VV.tif']
        else:
            self.path_list_s2 = ['Sentinel2_2020/S2_R1_MT_2020_08_03_2020_08_15_B1_B2-005.tif',
                'Sentinel2_2020/S2_R1_MT_2020_08_03_2020_08_15_B3_B4-011.tif',
                'Sentinel2_2020/S2_R1_MT_2020_08_03_2020_08_15_B5_B6-010.tif',
                'Sentinel2_2020/S2_R1_MT_2020_08_03_2020_08_15_B7_B8-012.tif',
                'Sentinel2_2020/S2_R1_MT_2020_08_03_2020_08_15_B8A_B9-007.tif',
                'Sentinel2_2020/S2_R1_MT_2020_08_03_2020_08_15_B10_B11-004.tif',
                'Sentinel2_2020/S2_R1_MT_2020_08_03_2020_08_15_B12.tif']

            self.path_list_s2_cloudy = ['Sentinel2_2020_Clouds/S2CL_R1_MT_2020_09_15_2020_09_18_B1_B2-003.tif',
                'Sentinel2_2020_Clouds/S2CL_R1_MT_2020_09_15_2020_09_18_B3_B4-005.tif',
                'Sentinel2_2020_Clouds/S2CL_R1_MT_2020_09_15_2020_09_18_B5_B6-006.tif',
                'Sentinel2_2020_Clouds/S2CL_R1_MT_2020_09_15_2020_09_18_B7_B8-004.tif',
                'Sentinel2_2020_Clouds/S2CL_R1_MT_2020_09_15_2020_09_18_B8A_B9-002.tif',
                'Sentinel2_2020_Clouds/S2CL_R1_MT_2020_09_15_2020_09_18_B10_B11-007.tif',
                'Sentinel2_2020_Clouds/S2CL_R1_MT_2020_09_15_2020_09_18_B12.tif']

            self.path_list_s1 = ['S1_R1_MT_2020_08_03_2020_08_08_VH.tif',
                'S1_R1_MT_2020_08_03_2020_08_08_VV.tif']

        super().__init__(date, crop_sample_im, normalize, loadIms)        
    def loadOptical(self, path_list, folder):
        s2_b1_2 = self.loadImage(self.root_path + path_list[0])[:, :-4000, 3000:]
        ic(s2_b1_2.shape, s2_b1_2.dtype)
        s2_b3_4 = self.loadImage(self.root_path + path_list[1])[:, :-4000, 3000:]
        s2_b5_6 = self.loadImage(self.root_path + path_list[2])[:, :-4000, 3000:]
        s2_b7_8 = self.loadImage(self.root_path + path_list[3])[:, :-4000, 3000:]
        s2_b8A_9 = self.loadImage(self.root_path + path_list[4])[:, :-4000, 3000:]
        s2_b10_11 = self.loadImage(self.root_path + path_list[5])[:, :-4000, 3000:]
        s2_b12 = self.loadImage(self.root_path + path_list[6])[:, :-4000, 3000:]

        ic(s2_b1_2.shape, s2_b3_4.shape, s2_b5_6.shape, s2_b7_8.shape, s2_b8A_9.shape)

        #pdb.set_trace()
        s2 = np.concatenate((s2_b1_2, s2_b3_4, s2_b5_6, s2_b7_8, s2_b8A_9, s2_b10_11, s2_b12), axis=0)
        del s2_b1_2, s2_b3_4, s2_b5_6, s2_b7_8, s2_b8A_9, s2_b10_11, s2_b12

        ic(s2.shape, s2.dtype)

    #    super().loadOptical(s2)
    
    #def loadOptical(self, s2):

        print(s2.shape)

        s2[s2 < 0] = 0
        s2[s2 > 10000] = 10000
        #np.save('s2_2018.npy', s2)

        print(np.min(s2[1]), np.average(s2[1]), np.max(s2[1]))
        print(np.min(s2[2]), np.average(s2[2]), np.max(s2[2]))
        print(np.min(s2[3]), np.average(s2[3]), np.max(s2[3]))
        
        #pdb.set_trace()
        return s2
    def loadMask(self):
        super().loadMask()
        self.mask = self.mask
        ic(self.mask.shape)
#        self.mask = self.mask[:-4000, 3000:]

class ImageNRW(Image):
    def __init__(self, date, crop_sample_im, normalize, loadIms, root_path):
        self.site = 'NRW'
        self.dim = (10980, 10980)
        self.root_path = root_path + self.site + '/'
        self.resolution_list = ['60m', '10m', '10m', '10m', '20m', '20m', '20m', '10m', 
                            '20m', '60m', '10m', '20m', '20m']

        self.mask_filename = 'mask_training_0_val_1_test_2.npy'
        self.path_list_s2 = ['R60m/T32UMC_20200601T103629_B01_60m.jp2',
                            'R10m/T32UMC_20200601T103629_B02_10m.jp2',
                            'R10m/T32UMC_20200601T103629_B03_10m.jp2',
                            'R10m/T32UMC_20200601T103629_B04_10m.jp2',
                            'R20m/T32UMC_20200601T103629_B05_20m.jp2',
                            'R20m/T32UMC_20200601T103629_B06_20m.jp2',
                            'R20m/T32UMC_20200601T103629_B07_20m.jp2',
                            'R10m/T32UMC_20200601T103629_B08_10m.jp2',
                            'R20m/T32UMC_20200601T103629_B8A_20m.jp2',
                            'R60m/T32UMC_20200601T103629_B09_60m.jp2',
                            'R10m/S2_NRW_2020_06_01_B10.tif',
                            'R20m/T32UMC_20200601T103629_B11_20m.jp2',
                            'R20m/T32UMC_20200601T103629_B12_20m.jp2']
        self.path_list_s2_cloudy = ['R60m/T32UMC_20200606T104031_B01_60m.jp2',
                            'R10m/T32UMC_20200606T104031_B02_10m.jp2',
                            'R10m/T32UMC_20200606T104031_B03_10m.jp2',
                            'R10m/T32UMC_20200606T104031_B04_10m.jp2',
                            'R20m/T32UMC_20200606T104031_B05_20m.jp2',
                            'R20m/T32UMC_20200606T104031_B06_20m.jp2',
                            'R20m/T32UMC_20200606T104031_B07_20m.jp2',
                            'R10m/T32UMC_20200606T104031_B08_10m.jp2',
                            'R20m/T32UMC_20200606T104031_B8A_20m.jp2',
                            'R60m/T32UMC_20200606T104031_B09_60m.jp2',
                            'R10m/S2_NRW_2020_06_06_B10.tif',
                            'R20m/T32UMC_20200606T104031_B11_20m.jp2',
                            'R20m/T32UMC_20200606T104031_B12_20m.jp2']

        self.path_list_s1 = ['S1/S1_NRW_2020_06_01_06_04_VH.tif', 
                            'S1/S1_NRW_2020_06_01_06_04_VV.tif']
        
        super().__init__(date, crop_sample_im, normalize, loadIms) 
    
    def loadOptical(self, path_list, folder):
        bands = []
        for path, resolution in zip(path_list, self.resolution_list):
            print("Starting band...")
            ic(resolution)
            band = loadImage(self.root_path + folder + path)
            ic(band.shape)
            if resolution == '20m' or resolution == '60m':
                band = cv2.resize(band, self.dim, interpolation = cv2.INTER_NEAREST)
            ic(band.shape)
            bands.append(band)
        
        s2 = np.stack(bands, axis = 0)

        s2[s2 < 0] = 0
        s2[s2 > 10000] = 10000

        ic(np.min(s2[1]), np.average(s2[1]), np.max(s2[1]))
        ic(np.min(s2[2]), np.average(s2[2]), np.max(s2[2]))
        ic(np.min(s2[3]), np.average(s2[3]), np.max(s2[3]))
        
        #pdb.set_trace()
        return s2