import numpy as np
from icecream import ic
import rasterio
from tools.image_metrics import metrics_get_mask, metrics_get
import pdb
from predictAmazon import Image

def get_raw_data(path):
    with rasterio.open(path, driver='GTiff') as src:
        image = src.read()

    # checkimage for nans
    image[np.isnan(image)] = np.nanmean(image)

    return image.astype('float32')

date = '2018'

true_filename = 's2_'+date+'.npy'
predictions_filename = 'predictions_scratch_unnorm_'+date+'.npy'
root_path = "D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/10m_all_bands/"
mask_filename = 'tile_mask_0tr_1vl_2ts.npy'

#s1 = get_raw_data(root_path + 'COPERNICUS_S1_20180719_20180726_VH.tif')
#target = np.load(true_filename)
crop_sample_im = False
im = Image(date = date, crop_sample_im = crop_sample_im)
#im.loadMask()

#pdb.set_trace()
predictions = np.load(predictions_filename)
mask = np.load(root_path + mask_filename)

ic(predictions.shape, im.s2.shape, mask.shape)
ic(predictions[...,:-3].shape, im.s2[...,:-3].shape, mask.shape)
#pdb.set_trace()
metrics_get(im.s2, predictions)
#metrics_get_mask(im.s2[...,:-3], predictions[...,:-3], mask)