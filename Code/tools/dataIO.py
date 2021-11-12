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
        ic(np.min(HH), np.average(HH), np.std(HH),np.max(HH))

        HH = np.clip(HH, -25.0, 0)
        HH = (HH + 25.1) * 255 / 25.1
        HV = np.clip(HV, -32.5, 0)
        HV = (HV + 32.6) * 255 / 32.6
        ic(np.min(HH), np.average(HH), np.std(HH),np.max(HH))

        rgb = np.dstack((np.zeros_like(HH), HH, HV))
        pdb.set_trace()
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


def generate_output_images(predicted, ID, predicted_images_path, 
    input_data_folder, cloud_threshold, remove_60m_bands):
    scene_name, filepath_sar, filepath_cloudFree, filepath_cloudy = get_info_quartet(ID,
                                                                                     predicted_images_path,
                                                                                     input_data_folder)

    ##ic(ID, filepath_sar, filepath_cloudFree)

    print("Generating quartet for ", scene_name)

    sar_preview = get_preview(filepath_sar, False, [1, 2, 2], sar_composite=True)
    if remove_60m_bands == False:
        opt_bands = [4, 3, 2]  # R, G, B bands (S2 channel numbers)
    else:
        opt_bands = [3, 2, 1]  # R, G, B bands (S2 channel numbers)

    cloudFree_preview = get_preview(filepath_cloudFree, False, opt_bands, brighten_limit=2000)
    cloudy_preview = get_preview(filepath_cloudy, False, opt_bands)
    cloudy_preview_brightened = get_preview(filepath_cloudy, False, opt_bands, brighten_limit=2000)

    predicted_preview = get_preview(predicted, True, opt_bands, 2000)

    ##ic(predicted.shape, filepath_cloudFree)
    ##ic(np.min(predicted_preview), np.average(predicted_preview), np.max(predicted_preview))
    ##ic(np.min(cloudFree_preview), np.average(cloudFree_preview), np.max(cloudFree_preview))
    
    ##ic(np.average(cloudFree_preview[-30:-1, -30:-1]), 
    ##    np.average(predicted_preview[-30:-1, -30:-1]),
    ##    np.average(cloudy_preview[-30:-1, -30:-1]))
    cloud_mask = get_cloud_cloudshadow_mask(get_raw_data(filepath_cloudy), cloud_threshold)
    save_single_images(sar_preview, cloudy_preview, cloudFree_preview, predicted_preview, cloudy_preview_brightened,
                       cloud_mask, predicted_images_path, scene_name)
##    pdb.set_trace()
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


def save_single_images(sar_preview, cloudy_preview, cloudFree_preview, predicted_preview, cloudy_preview_brightened,
                       cloud_mask, predicted_images_path, scene_name):
    out_path = make_dir(os.path.join(predicted_images_path, scene_name))

    ##ic(predicted_preview.shape, cloudFree_preview.shape)
#    pdb.set_trace()
    save_single_image(sar_preview, out_path, "inputsar")
    save_single_image(cloudy_preview, out_path, "input")
    save_single_image(cloudFree_preview, out_path, "inputtarg")
    save_single_image(predicted_preview, out_path, "inputpred")
    save_single_image(cloudy_preview_brightened, out_path, "inputbr")
    save_single_cloudmap(cloud_mask, out_path, "cloudmask")

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
class DataGenerator(keras.utils.Sequence):
    """DataGenerator for Keras routines."""

    def __init__(self,
                 list_IDs,
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

        if clip_min is None:
            clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.remove_60m_bands = remove_60m_bands
        self.input_dim = input_dim
        self.batch_size = batch_size
        ic(list_IDs)
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
        ##ic(self.indexes.shape, index, self.batch_size)
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        ##ic(indexes)
        ##ic(len(self.list_IDs), self.list_IDs[0])
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        if self.include_target:
            # Generate data
            X, y = self.__data_generation(list_IDs_temp, self.augment_rotation_param[indexes],
                                          self.augment_flip_param[indexes], self.random_crop_paramx[indexes],
                                          self.random_crop_paramy[indexes])
            return X, y
        else:
            X = self.__data_generation(list_IDs_temp, self.augment_rotation_param[indexes],
                                       self.augment_flip_param[indexes], self.random_crop_paramx[indexes],
                                       self.random_crop_paramy[indexes])
            return X

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

        #pdb.set_trace()
        if self.include_target:
            output_opt_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp,
                                              random_crop_paramx_temp, random_crop_paramy_temp, data_type=2)
            # print("========")
            # print("hi",np.min(input_sar_batch), 
            #     np.average(input_sar_batch), 
            #     np.std(input_sar_batch), 
            #     np.max(input_sar_batch))
            # print("========")

            if self.use_cloud_mask > 0:
                output_opt_cloud_batch = [np.append(output_opt_batch[sample], cloud_mask_batch[sample], axis=0) for
                                          sample in range(len(output_opt_batch))]
                output_opt_cloud_batch = np.asarray(output_opt_cloud_batch)
                #print("HEre!!")
                #ic(input_opt_batch.shape, input_sar_batch.shape, output_opt_cloud_batch.shape)
                #ic(output_opt_batch.shape)
                #pdb.set_trace()

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
            ic(np.min(data_image[0]), np.average(data_image[0]), np.std(data_image[0]), np.max(data_image[0]))
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (
                        self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
            print("S1")
            print(np.min(data_image[0]), np.average(data_image[0]), np.std(data_image[0]), np.max(data_image[0]))

            if shift_data:
                data_image -= self.max_val / 2
            print(np.min(data_image[0]), np.average(data_image[0]), np.std(data_image[0]), np.max(data_image[0]))
            
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

        batch = np.empty((self.batch_size, *dim)).astype('float32')
        ##ic(batch.shape)
        cloud_mask_batch = np.empty((self.batch_size, self.input_dim[0][1], self.input_dim[0][2])).astype('float32')
        #ic(list_IDs_temp)
        for i, ID in enumerate(list_IDs_temp):
            #ic(ID)
            data_image = self.get_data_image(ID, data_type, random_crop_paramx_temp[i], random_crop_paramy_temp[i])
            #ic(data_image.shape)
            # pdb.set_trace()

            if self.data_augmentation:
                if not augment_flip_param_temp[i] == 0:
                    data_image = np.flip(data_image, augment_flip_param_temp[i])
                if not augment_rotation_param_temp[i] == 0:
                    data_image = np.rot90(data_image, augment_rotation_param_temp[i], (1, 2))

            if data_type == 3 and self.use_cloud_mask:
                cloud_mask = get_cloud_cloudshadow_mask(data_image, self.cloud_threshold)
                #cloud_mask[cloud_mask != 0] = 1
                cloud_mask[cloud_mask != 1] = 0
                
                ic(i, ID, cloud_mask.shape, data_image.shape)
                cloud_mask_batch[i,] = cloud_mask

            if data_type == 2 or data_type == 3:
                if self.remove_60m_bands == True:
                    #ic(np.r_[1:9,11:13])
                    data_image = data_image[np.r_[1:9,11:13]]
                    #ic(data_image.shape)
                    #pdb.set_trace()
            
            data_image = self.get_normalized_data(data_image, data_type)

            batch[i,] = data_image

        cloud_mask_batch = cloud_mask_batch[:, np.newaxis, :, :]

        if data_type == 3:
            return batch, cloud_mask_batch
        else:
            return batch

# %%



def GeoReference_Raster_from_Source_data(source_file, 
    numpy_image, target_file, bands):

    with rasterio.open(source_file) as src:
        ras_meta = src.profile

    ras_meta.update(count=bands)

    with rasterio.open(target_file, 'w', **ras_meta) as dst:
        dst.write(numpy_image)

def Split_in_Patches(rows, cols, patch_size, mask, 
                     augmentation_list, 
                     prefix=0, percent=0):

    """
    Everything  in this function is made operating with
    the upper left corner of the patch
    mask: tr, vl, ts
    lbl: labels
    prefix: refers to a certain image (t0 or t1)
    percent: overlap between consecutive patches
    """

    # Percent of overlap between consecutive patches.
    overlap = round(patch_size * percent)
    overlap -= overlap % 2
    stride = patch_size - overlap
    # Add Padding to the image to match with the patch size
    step_row = (stride - rows % stride) % stride
    step_col = (stride - cols % stride) % stride
    pad_tuple_msk = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)) )
    # lbl = np.pad(lbl, pad_tuple_msk, mode = 'symmetric')
    mask_pad = np.pad(mask, pad_tuple_msk, mode = 'symmetric')
    # cloud_mask = np.pad(cloud_mask, pad_tuple_msk, mode = 'symmetric')

    k1, k2 = (rows+step_row)//stride, (cols+step_col)//stride
    print('Total number of patches: %d x %d' %(k1, k2))

    train_mask = np.zeros_like(mask_pad)
    val_mask = np.zeros_like(mask_pad)
    test_mask = np.zeros_like(mask_pad)
    train_mask[mask_pad==0] = 1
    test_mask [mask_pad==2] = 1
    val_mask = (1-train_mask) * (1-test_mask)

    train_patches, val_patches, test_patches = [], [], []
    only_bck_patches = 0
    cloudy_patches = 0
    # lbl[lbl!=1] = 0
    for i in range(k1):
        for j in range(k2):
            # Train
            if train_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                ## if cloud_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
                ##     cloudy_patches += 1
                ##     continue
                for k in augmentation_list:
                    train_patches.append((prefix, i*stride, j*stride, k))
                ## if not lbl[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
                    # train_patches.append((prefix, i*stride, j*stride, 0))
                ##    only_bck_patches += 1
            # Test                !!!!!Not necessary with high overlap!!!!!!!!
            elif test_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                test_patches.append((prefix, i*stride, j*stride, 0))
            # Val                 !!!!!Not necessary with high overlap!!!!!!!!
            elif val_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                val_patches.append((prefix, i*stride, j*stride, 0))
    ## print('Training Patches with background only: %d' %(only_bck_patches))
    ## print('Patches with clouds in the cloud-free image: %d' %(cloudy_patches))
    
    return train_patches, val_patches, test_patches, step_row, step_col, overlap