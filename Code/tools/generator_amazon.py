import csv
import os
import os.path

import keras
import matplotlib
import numpy as np
import rasterio
import scipy.signal as scisig
from matplotlib import pyplot as plt
from tools.feature_detectors import get_cloud_cloudshadow_mask

from icecream import ic
import pdb

ic.configureOutput(includeContext=True)

def make_dir(dir_path):
    if os.path.isdir(dir_path):
        print("WARNING: Folder {} exists and content may be overwritten!")
    else:
        os.makedirs(dir_path)

    return dir_path


def get_train_val_test_filelists(listpath):
    with open(listpath) as f:
        reader = csv.reader(f, delimiter='\t')
        filelist = list(reader)
    ##ic(filelist)
    train_filelist = []
    val_filelist = []
    test_filelist = []
    for f in filelist:
        line_entries = f[0].split(sep=", ")
        ##ic(line_entries)
        if line_entries[0] == '1':
            train_filelist.append(line_entries)
        if line_entries[0] == '2':
            val_filelist.append(line_entries)
        if line_entries[0] == '3':
            test_filelist.append(line_entries)
    
    ##ic(train_filelist)
    ##ic(val_filelist)
#    pdb.set_trace()
    return train_filelist, val_filelist, test_filelist


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Output%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Output%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Output%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def get_info_quartet(ID, predicted_images_path, input_data_folder):
    ##ic(ID)
#    pdb.set_trace()
#    scene_name = ID[4]
    scene_name = ID.split('/')[-1]
    filepath_sar = os.path.join(input_data_folder, ID.replace('sensor', 's1'))
    filepath_cloudFree = os.path.join(input_data_folder, ID.replace('sensor', 's2'))
    filepath_cloudy = os.path.join(input_data_folder, ID.replace('sensor', 's2_cloudy'))

#    filepath_sar = os.path.join(input_data_folder, ID[1], ID[4]).lstrip()
#    filepath_cloudFree = os.path.join(input_data_folder, ID[2], ID[4]).lstrip()
#    filepath_cloudy = os.path.join(input_data_folder, ID[3], ID[4]).lstrip()

    return scene_name[:-4], filepath_sar, filepath_cloudFree, filepath_cloudy


def get_rgb_preview(r, g, b, sar_composite=False):
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
        #ic(np.min(HH), np.average(HH), np.std(HH),np.max(HH))

        HH = np.clip(HH, -25.0, 0)
        HH = (HH + 25.1) * 255 / 25.1
        HV = np.clip(HV, -32.5, 0)
        HV = (HV + 32.6) * 255 / 32.6
        #ic(np.min(HH), np.average(HH), np.std(HH),np.max(HH))

        rgb = np.dstack((np.zeros_like(HH), HH, HV))
        #pdb.set_trace()
        return rgb.astype(np.uint8)


def get_raw_data(path):
    with rasterio.open(path, driver='GTiff') as src:
        image = src.read()

    # checkimage for nans
    image[np.isnan(image)] = np.nanmean(image)

    return image.astype('float32')


def get_preview(file, predicted_file, bands, brighten_limit=None, sar_composite=False):
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
        return get_rgb_preview(r, g, b, sar_composite)
    else:
        r = np.clip(r, 0, brighten_limit)
        g = np.clip(g, 0, brighten_limit)
        b = np.clip(b, 0, brighten_limit)
        return get_rgb_preview(r, g, b, sar_composite)

def saveSampleIms(self, s1, s2, s2_cloudy, predicted):
    # predicted *= self.scale

    generate_output_images(s1, s2, s2_cloudy, predicted * 2000)

def generate_output_images( s1, s2, s2_cloudy, predicted, 
    predicted_images_path = 'sample_ims', scene_name = '2018'):

    sar_preview = get_preview(s1, True, [1, 2, 2], sar_composite=False)

    opt_bands = [4, 3, 2]  # R, G, B bands (S2 channel numbers)

    cloudFree_preview = get_preview(s2, True, opt_bands, brighten_limit=2000)
    cloudy_preview = get_preview(s2_cloudy, True, opt_bands)
    
    predicted_preview = get_preview(predicted, True, opt_bands, 2000)

    # ic(np.min(predicted_preview), np.average(predicted_preview), np.max(predicted_preview))

    #predicted_images_path = 'sample_ims'
    #scene_name = '2018'
    out_path = make_dir(os.path.join(predicted_images_path, scene_name))

    save_single_images(sar_preview, cloudy_preview, cloudFree_preview, 
        predicted_preview, out_path)

    return
    
def save_single_image(image, out_path, name):
    plt.figure(frameon=False)
    plt.imshow(image)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.axis('off')
    plt.savefig(os.path.join(out_path, name + '.png'), dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def save_single_cloudmap(image, out_path, name):
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'white'])

    bounds = [-1, -0.5, 0.5, 1]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    plt.figure()
    plt.imshow(image, cmap=cmap, norm=norm, vmin=-1, vmax=1)

    cb = plt.colorbar(aspect=40, pad=0.01)
    cb.ax.yaxis.set_tick_params(pad=0.9, length=2)

    cb.ax.yaxis.set_ticks([0.33 / 2, 0.5, 1 - 0.33 / 2])
    cb.ax.yaxis.set_ticklabels(['Shadow', 'Clear', 'Cloud'])

    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.savefig(os.path.join(out_path, name + '.png'), dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def save_single_images(sar_preview, cloudy_preview, cloudFree_preview, predicted_preview, 
    out_path):

#, cloudy_preview_brightened,
#                       cloud_mask, predicted_images_path, scene_name):
#    out_path = make_dir(os.path.join(predicted_images_path, scene_name))

    ##ic(predicted_preview.shape, cloudFree_preview.shape)
#    pdb.set_trace()
    save_single_image(sar_preview, out_path, "inputsar")
    save_single_image(cloudy_preview, out_path, "input")
    save_single_image(cloudFree_preview, out_path, "inputtarg")
    save_single_image(predicted_preview, out_path, "inputpred")
#    save_single_image(cloudy_preview_brightened, out_path, "inputbr")
#    save_single_cloudmap(cloud_mask, out_path, "cloudmask")

    return


def process_predicted(predicted, ID, predicted_images_path, scale, 
    cloud_threshold, input_data_folder, remove_60m_bands):
    ##ic(ID)
    ##ic(predicted.shape)
    for i, data_image in enumerate(predicted):
        data_image *= scale
        ##ic(i)
        generate_output_images(data_image, ID[i], predicted_images_path, 
            input_data_folder, cloud_threshold, remove_60m_bands)

    return


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Input%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Input%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Input%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class DataGeneratorAmazon(keras.utils.Sequence):
    """DataGenerator for Keras routines."""

    def __init__(self,
                 list_IDs,
                 ims,
                 batch_size=32,
                 input_dim=((13, 256, 256), (2, 256, 256)),
                 scale=2000,
                 shuffle=True,
                 include_target=True,
                 data_augmentation=False,
                 random_crop=False,
                 crop_size=128,
                 clip_min=None,
                 clip_max=None,
                 input_data_folder='./',
                 use_cloud_mask=True,
                 max_val_sar=5,
                 cloud_threshold=0.2,
                 remove_60m_bands = False
                 ):
        self.ims = ims

        if clip_min is None:
            clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.remove_60m_bands = remove_60m_bands
        self.input_dim = input_dim
        self.batch_size = batch_size
        # ic(list_IDs)
        self.list_IDs = list_IDs
        self.nr_images = len(self.list_IDs)
        self.indexes = np.arange(self.nr_images)
        self.scale = scale
        self.shuffle = shuffle
        self.include_target = include_target
        self.data_augmentation = data_augmentation
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.max_val = max_val_sar

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.input_data_folder = input_data_folder
        self.use_cloud_mask = use_cloud_mask
        self.cloud_threshold = cloud_threshold

        self.augment_rotation_param = np.repeat(0, self.nr_images)
        self.augment_flip_param = np.repeat(0, self.nr_images)
        self.random_crop_paramx = np.repeat(0, self.nr_images)
        self.random_crop_paramy = np.repeat(0, self.nr_images)

        
#        ic(self.random_crop_paramx)
#        pdb.set_trace()
        self.on_epoch_end()

        print("Generator initialized")
        ##ic(self.batch_size)


    def __len__(self):
        """Gets the number of batches per epoch"""
        return int(np.floor(self.nr_images / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch from shuffled indices list
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]


        # Generate data
        X, y = self.__data_generation(list_IDs_temp, self.augment_rotation_param[indexes],
                                        self.augment_flip_param[indexes], self.random_crop_paramx[indexes],
                                        self.random_crop_paramy[indexes])
        return X, y

    def on_epoch_end(self):
        """Update indexes after each epoch."""

        if self.shuffle:
            np.random.shuffle(self.indexes)

        if self.data_augmentation:
            self.augment_rotation_param = np.random.randint(0, 4, self.nr_images)
            self.augment_flip_param = np.random.randint(0, 3, self.nr_images)

        if self.random_crop:
            self.random_crop_paramx = np.random.randint(0, self.crop_size, self.nr_images)
            self.random_crop_paramy = np.random.randint(0, self.crop_size, self.nr_images)
        return

    def __data_generation(self, list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp,
                          random_crop_paramx_temp, random_crop_paramy_temp):

        input_opt_batch, cloud_mask_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp,
                                                           augment_flip_param_temp, random_crop_paramx_temp,
                                                           random_crop_paramy_temp, data_type=3)

        input_sar_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp,
                                         random_crop_paramx_temp, random_crop_paramy_temp, data_type=1)

        #plt.figure()
        #plt.imshow(np.transpose(input_opt_batch[1:4], (1, 2, 0)))
        #plt.savefig('s2_cloudy_patch.png')
        #pdb.set_trace()    
        # print("1", input_opt_batch.shape, input_sar_batch.shape)
        if self.include_target:
            output_opt_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp,
                                              random_crop_paramx_temp, random_crop_paramy_temp, data_type=2)
            # print("2", input_opt_batch.shape, input_sar_batch.shape, output_opt_batch.shape)
            
            # print("========")
            # print("hi",np.min(input_sar_batch), 
            #     np.average(input_sar_batch), 
            #     np.std(input_sar_batch), 
            #     np.max(input_sar_batch))
            # print("========")

            
            # print(np.average(input_opt_batch), np.average(output_opt_batch))
            # print(np.average(input_opt_batch[0]), np.average(output_opt_batch[0]))
            # print(np.average(input_opt_batch[1]), np.average(output_opt_batch[1]))
            # np.save('output_opt_batch.npy', output_opt_batch)
            # np.save('input_opt_batch.npy', input_opt_batch)
            # generate_output_images(input_sar_batch[0], output_opt_batch[0], input_opt_batch[0], output_opt_batch[0])

            '''
            print("=====*====")
            print(np.average(input_opt_batch), np.average(output_opt_batch))
            print(np.average(input_opt_batch[0]), np.average(output_opt_batch[0]))
            print(np.average(input_opt_batch[1]), np.average(output_opt_batch[1]))
            
            
            print("=====*====")
            pdb.set_trace()
            #plt.figure()
            #plt.imshow(np.transpose(output_opt_batch[1:4], (1, 2, 0)))
            #plt.savefig('s2_patch.png')
            '''
            
            if self.use_cloud_mask > 0:
                output_opt_cloud_batch = [np.append(output_opt_batch[sample], cloud_mask_batch[sample], axis=0) for
                                          sample in range(len(output_opt_batch))]
                output_opt_cloud_batch = np.asarray(output_opt_cloud_batch)
                #print("HEre!!")
                #ic(input_opt_batch.shape, input_sar_batch.shape, output_opt_cloud_batch.shape)
                #ic(output_opt_batch.shape)
                #pdb.set_trace()
                # print("s2_cloudy generator",np.average(input_opt_batch))
                # print("s2_cloudy generator2",np.average(self.s1_cloudy_t0))

                return ([input_opt_batch, input_sar_batch], [output_opt_cloud_batch])
            else:
                return ([input_opt_batch, input_sar_batch], [output_opt_batch])
        elif not self.include_target:
            # for prediction step where target is predicted
            return ([input_opt_batch, input_sar_batch])

    def get_image_data(self, paramx, paramy, path):
        # with block not working with window kw
        src = rasterio.open(path, 'r', driver='GTiff')
        ##ic(paramx, paramy, self.crop_size)
        #paramx = 0
        #paramy = 0
        #paramx = self.random_crop_paramx_
        #paramy = self.random_crop_paramx_

        image = src.read(window=((paramx, paramx + self.crop_size), (paramy, paramy + self.crop_size)))
        src.close()
        image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts
#        if image.shape[1] == 41 or image.shape[1] == 95:
#            ic(path, image.shape)
#            pdb.set_trace()

        return image

    def get_opt_image(self, path, paramx, paramy):

        image = self.get_image_data(paramx, paramy, path)

        return image.astype('float32')

    def get_sar_image(self, path, paramx, paramy):

        image = self.get_image_data(paramx, paramy, path)

        medianfilter_onsar = False
        if medianfilter_onsar:
            image[0] = scisig.medfilt2d(image[0], 7)
            image[1] = scisig.medfilt2d(image[1], 7)

        return image.astype('float32')

    def get_data_image(self, ID, data_type, paramx, paramy):
        ##ic(ID)
        ##ic(self.input_data_folder)
        sensors = ['s1', 's2', 's2_cloudy']
#        data_path = os.path.join(self.input_data_folder, ID[data_type], ID[4]).lstrip()
        # ic(ID)
        # pdb.set_trace()
        ID = ID.replace('sensor', sensors[data_type - 1])
        data_path = os.path.join(self.input_data_folder, ID).lstrip()

#        data_path = ID.lstrip()

        if data_type == 2 or data_type == 3:
            data_image = self.get_opt_image(data_path, paramx, paramy)

        elif data_type == 1:
            data_image = self.get_sar_image(data_path, paramx, paramy)
        else:
            print('Error! Data type invalid')

        return data_image

    def get_normalized_data(self, data_image, data_type):

        shift_data = False

        shift_values = [[0, 0], [1300., 981., 810., 380., 990., 2270., 2070., 2140., 2200., 650., 15., 1600., 680.],
                        [1545., 1212., 1012., 713., 1212., 2476., 2842., 2775., 3174., 546., 24., 1776., 813.]]

        # SAR
        if data_type == 1:
            #print(np.min(data_image[0]), np.average(data_image[0]), np.std(data_image[0]), np.max(data_image[0]))

            #ic(np.min(data_image[0]), np.average(data_image[0]), np.std(data_image[0]), np.max(data_image[0]))
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (
                        self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
            #print("S1")
            #print(np.min(data_image[0]), np.average(data_image[0]), np.std(data_image[0]), np.max(data_image[0]))

            if shift_data:
                data_image -= self.max_val / 2
            #
            #print(np.min(data_image[0]), np.average(data_image[0]), np.std(data_image[0]), np.max(data_image[0]))

        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
                if shift_data:
                    data_image[channel] -= shift_values[data_type - 1][channel]

            data_image /= self.scale

        return data_image

    def get_batch(self, list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp, random_crop_paramx_temp,
                  random_crop_paramy_temp, data_type):
        
        if data_type == 1:
            dim = self.input_dim[1]
        else:
            dim = self.input_dim[0]

        if data_type == 1:
            im = [self.ims['s1_t0'], self.ims['s1_t1']] 
        elif data_type == 2:
            im = [self.ims['s2_t0'], self.ims['s2_t1']]
        elif data_type == 3:
            im = [self.ims['s2_cloudy_t0'], self.ims['s2_cloudy_t1']]

        batch = np.empty((self.batch_size, *dim)).astype('float32')
        ##ic(batch.shape)
        cloud_mask_batch = np.empty((self.batch_size, self.input_dim[0][1], self.input_dim[0][2])).astype('float32')
        #ic(list_IDs_temp)
        for i, ID in enumerate(list_IDs_temp):
            # print("ID", ID)
            # if data_type == 3:
            #     print("Before slice", np.shape(im), np.average(im), np.average(self.s1_cloudy_t0))
            data_image = im[ID[0]][:,ID[1]:ID[1]+self.crop_size,
                          ID[2]:ID[2]+self.crop_size].copy()
            # if data_type == 3:
            #     print("After slice", np.shape(data_image), np.average(data_image))
            '''
            if data_type == 3:
                print("=====")
                print("i",i)
                print("ID", ID)
                print("im.shape", im[0].shape)
                print(np.average(self.s1_cloudy_t0[:,ID[1]:ID[1]+self.crop_size,
                          ID[2]:ID[2]+self.crop_size]))
                print("np.average(data_image)", np.average(data_image))
                print("=====")

                # pdb.set_trace()
            '''

            '''
            if data_image.shape[1] == 39:
                print("data image shape", data_image.shape)
                pdb.set_trace()
            print(data_image.shape, "hi", self.crop_size, ID, im[ID[0]].shape)
            '''

                    

            
            if self.data_augmentation:
                if not augment_flip_param_temp[i] == 0:
                    data_image = np.flip(data_image, augment_flip_param_temp[i])
                if not augment_rotation_param_temp[i] == 0:
                    data_image = np.rot90(data_image, augment_rotation_param_temp[i], (1, 2))


            if data_type == 3 and self.use_cloud_mask:
                cloud_mask = get_cloud_cloudshadow_mask(data_image, self.cloud_threshold)
                cloud_mask[cloud_mask != 0] = 1
                ## print(i, ID, cloud_mask.shape, data_image.shape)
                #if cloud_mask.shape != (128,128):
                #    print("Bad shape", cloud_mask.shape)
                #assert cloud_mask.shape == (128,128), cloud_mask.shape
                # ic(i, ID, cloud_mask.shape, data_image.shape)
                '''
                print("data image shape", data_image.shape)
                assert data_image.shape[1] != 39, "Error in shape" + str(data_image.shape[1]) + " " + str(ID[0]) \
                    + " " + str(ID[0]) \
                    + " " + str(ID[1]) \
                    + " " + str(ID[2]) \
                    + " " + str(self.crop_size) 
                '''

                #if cloud_mask.shape == (128,128):
                cloud_mask_batch[i,] = cloud_mask
                #else:
                #    print("Bad cloud mask shape", cloud_mask.shape)


            if data_type == 2 or data_type == 3:
                if self.remove_60m_bands == True:
                    #ic(np.r_[1:9,11:13])
                    data_image = data_image[np.r_[1:9,11:13]]
                    #ic(data_image.shape)
                    #pdb.set_trace()

            data_image = self.get_normalized_data(data_image, data_type)
            # if data_type == 3:
            #     print("After norm", np.shape(data_image), np.average(data_image))

            batch[i,] = data_image

        cloud_mask_batch = cloud_mask_batch[:, np.newaxis, :, :]

        if data_type == 3:
            return batch, cloud_mask_batch
        else:
            return batch
