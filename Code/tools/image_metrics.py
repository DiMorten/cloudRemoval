import keras.backend as K
import tensorflow as tf
from icecream import ic
import numpy as np
ic.enable()
from skimage.metrics import structural_similarity
import numpy as np

remove_60m_bands = True
if remove_60m_bands == False:
    s2_bands = 13
else:
    s2_bands = 10
# def metrics_get(inputs, y_true, y_pred): 
def metrics_get(y_true, y_pred):
    cloud_mean_absolute_error = np_cloud_mean_absolute_error(y_true, y_pred)
    cloud_mean_squared_error = np_cloud_mean_squared_error(y_true, y_pred)
    cloud_root_mean_squared_error = np_cloud_root_mean_squared_error(y_true, y_pred)

    cloud_psnr = np_cloud_psnr(y_true, y_pred)
    sam = np_cloud_mean_sam(y_true, y_pred)
    ic(cloud_mean_absolute_error,
        cloud_mean_squared_error,
        cloud_root_mean_squared_error,
        cloud_psnr,
        sam)#,
    # ssim = SSIM_large_image(y_true * 2000, y_pred * 2000)

    # ic(ssim)


def np_cloud_mean_absolute_error(y_true, y_pred):
    """Computes the MAE over the full image."""
    return np.mean(np.abs(y_pred[..., 0:s2_bands, :, :] - y_true[..., 0:s2_bands, :, :]))

def np_cloud_mean_squared_error(y_true, y_pred):
    """Computes the MSE over the full image."""
    return np.mean(np.square(y_pred[..., 0:s2_bands, :, :] - y_true[..., 0:s2_bands, :, :]))

def np_cloud_psnr(y_true, y_predict):
    """Computes the PSNR over the full image."""
#    y_true *= 2000
#    y_predict *= 2000
    rmse = np.sqrt(np.mean(np.square((y_predict * 2000)[..., 0:s2_bands, :, :] - (y_true * 2000)[..., 0:s2_bands, :, :])))

    return 20.0 * (np.log(10000.0 / rmse) / np.log(10.0))

def np_cloud_root_mean_squared_error(y_true, y_pred):
    """Computes the RMSE over the full image."""
    return np.sqrt(np.mean(np.square(y_pred[:, 0:s2_bands, :, :] - y_true[:, 0:s2_bands, :, :])))

# ==== Metrics for Amazon image
def np_cloud_mean_absolute_error(y_true, y_pred):
    """Computes the MAE over the full image."""
    return np.mean(np.abs(y_pred - y_true))


def np_cloud_mean_squared_error(y_true, y_pred):
    """Computes the MSE over the full image."""
    return np.mean(np.square(y_pred - y_true))

def np_cloud_psnr(y_true, y_predict):
    """Computes the PSNR over the full image."""
    #y_true *= 2000
    #y_predict *= 2000
    rmse = np.sqrt(np.mean(np.square(y_predict * 2000 - y_true * 2000)))

    return 20.0 * (np.log(10000.0 / rmse) / np.log(10.0))


def np_cloud_root_mean_squared_error(y_true, y_pred):
    """Computes the RMSE over the full image."""
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def np_get_sam(y_true, y_predict):
    """Computes the SAM array."""
    mat = np.multiply(y_true, y_predict)
    mat = np.sum(mat, 0)
    mat = np.divide(mat, np.sqrt(np.sum(np.multiply(y_true, y_true), 0)))
    mat = np.divide(mat, np.sqrt(np.sum(np.multiply(y_predict, y_predict), 0)))
    mat = np.arccos(np.clip(mat, -1, 1))

    return mat
def np_cloud_mean_sam(y_true, y_predict):
    """Computes the SAM over the full image."""
    mat = np_get_sam(y_true, y_predict)

    return np.mean(mat)

def SSIM(y_true, y_pred):
    """Computes the SSIM over the full image."""
    y_true = np.clip(y_true, 0, 10000.0)
    y_pred = np.clip(y_pred, 0, 10000.0)
    ssim = structural_similarity(y_true, y_pred, data_range=10000.0, multichannel = True)
    return ssim

#(...)
def SSIM_large_image(y_true, y_pred):
    ic(np.average(y_true), np.average(y_pred))
    sz = 3 # increase this number if there is not enough RAM
    rs, cs, _ = np.asarray(np.transpose(y_true,(1,2,0)).shape) // sz
    ic(rs, cs)
    ssim_0 = []
    for i in range(sz):
        
        i_ = rs*2 if i == sz-1 else rs
        for j in range(sz):
            ic(i,j)
            j_ = cs*2 if j == sz-1 else cs
            #ic(y_true.shape)
            #ic(np.transpose(y_true,(1,2,0))[i*rs:i*rs+i_, j*cs:j*cs+j_, :].shape)
            #ic(i, i*rs, i*rs+i_, j, j*cs, j*cs+j_)
            ssim_0.append(SSIM(np.transpose(y_true,(1,2,0))[i*rs:i*rs+i_, j*cs:j*cs+j_, :],
                np.transpose(y_pred,(1,2,0))[i*rs:i*rs+i_, j*cs:j*cs+j_, :]))

    return np.asarray(ssim_0).mean()

#==== end metrics for Amazon image

def cloud_mean_absolute_error(y_true, y_pred):
    """Computes the MAE over the full image."""
    print("shape", K.int_shape(y_pred), K.int_shape(y_true))
    print("shape", K.int_shape(y_pred[:, 0:s2_bands, :, :]), K.int_shape(y_true[:, 0:s2_bands, :, :]))
    
    return K.mean(K.abs(y_pred[:, 0:s2_bands, :, :] - y_true[:, 0:s2_bands, :, :]))


def cloud_mean_squared_error(y_true, y_pred):
    """Computes the MSE over the full image."""
    return K.mean(K.square(y_pred[:, 0:s2_bands, :, :] - y_true[:, 0:s2_bands, :, :]))


def cloud_root_mean_squared_error(y_true, y_pred):
    """Computes the RMSE over the full image."""
    return K.sqrt(K.mean(K.square(y_pred[:, 0:s2_bands, :, :] - y_true[:, 0:s2_bands, :, :])))


def cloud_bandwise_root_mean_squared_error(y_true, y_pred):
    return K.mean(K.sqrt(K.mean(K.square(y_pred[:, 0:s2_bands, :, :] - y_true[:, 0:s2_bands, :, :]), axis=[2, 3])))


def cloud_ssim(y_true, y_pred):
    """Computes the SSIM over the full image."""
    y_true = y_true[:, 0:s2_bands, :, :]
    y_pred = y_pred[:, 0:s2_bands, :, :]

    y_true *= 2000
    y_pred *= 2000

    y_true = tf.transpose(y_true, [0, 2, 3, 1])
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])

    ssim = tf.image.ssim(y_true, y_pred, max_val=10000.0)
    ssim = tf.reduce_mean(ssim)
    
    y_true /= 2000
    y_pred /= 2000

    return ssim


def get_sam(y_true, y_predict):
    """Computes the SAM array."""
    mat = tf.multiply(y_true, y_predict)
    mat = tf.reduce_sum(mat, 1)
    mat = tf.div(mat, K.sqrt(tf.reduce_sum(tf.multiply(y_true, y_true), 1)))
    mat = tf.div(mat, K.sqrt(tf.reduce_sum(tf.multiply(y_predict, y_predict), 1)))
    mat = tf.acos(K.clip(mat, -1, 1))

    return mat


def cloud_mean_sam(y_true, y_predict):
    """Computes the SAM over the full image."""
    mat = get_sam(y_true[:, 0:s2_bands, :, :], y_predict[:, 0:s2_bands, :, :])

    return tf.reduce_mean(mat)


def cloud_mean_sam_covered(y_true, y_pred):
    """Computes the SAM over the covered image parts."""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    target = y_true[:, 0:s2_bands, :, :]
    predicted = y_pred[:, 0:s2_bands, :, :]

    if K.sum(cloud_cloudshadow_mask) == 0:
        return 0.0

    sam = get_sam(target, predicted)
    sam = tf.expand_dims(sam, 1)
    sam = K.sum(cloud_cloudshadow_mask * sam) / K.sum(cloud_cloudshadow_mask)

    return sam


def cloud_mean_sam_clear(y_true, y_pred):
    """Computes the SAM over the clear image parts."""
    clearmask = K.ones_like(y_true[:, -1:, :, :]) - y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:s2_bands, :, :]
    input_cloudy = y_pred[:, -(s2_bands + 1):-1, :, :]

    if K.sum(clearmask) == 0:
        return 0.0

    sam = get_sam(input_cloudy, predicted)
    sam = tf.expand_dims(sam, 1)
    sam = K.sum(clearmask * sam) / K.sum(clearmask)

    return sam


def cloud_psnr(y_true, y_predict):
    """Computes the PSNR over the full image."""
    y_true *= 2000
    y_predict *= 2000
    rmse = K.sqrt(K.mean(K.square(y_predict[:, 0:s2_bands, :, :] - y_true[:, 0:s2_bands, :, :])))

    return 20.0 * (K.log(10000.0 / rmse) / K.log(10.0))


def cloud_mean_absolute_error_clear(y_true, y_pred):
    """Computes the SAM over the clear image parts."""
    clearmask = K.ones_like(y_true[:, -1:, :, :]) - y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:s2_bands, :, :]
    input_cloudy = y_pred[:, -(s2_bands + 1):-1, :, :]

    if K.sum(clearmask) == 0:
        return 0.0

    clti = clearmask * K.abs(predicted - input_cloudy)
    clti = K.sum(clti) / (K.sum(clearmask) * s2_bands)

    return clti


def cloud_mean_absolute_error_covered(y_true, y_pred):
    """Computes the SAM over the covered image parts."""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:s2_bands, :, :]
    target = y_true[:, 0:s2_bands, :, :]

    if K.sum(cloud_cloudshadow_mask) == 0:
        return 0.0

    ccmaec = cloud_cloudshadow_mask * K.abs(predicted - target)
    ccmaec = K.sum(ccmaec) / (K.sum(cloud_cloudshadow_mask) * s2_bands)

    return ccmaec


def carl_error(y_true, y_pred):
    """Computes the Cloud-Adaptive Regularized Loss (CARL)"""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    clearmask = K.ones_like(y_true[:, -1:, :, :]) - y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:s2_bands, :, :]
    input_cloudy = y_pred[:, -(s2_bands + 1):-1, :, :]
    target = y_true[:, 0:s2_bands, :, :]

    cscmae = K.mean(clearmask * K.abs(predicted - input_cloudy) + cloud_cloudshadow_mask * K.abs(
        predicted - target)) + 1.0 * K.mean(K.abs(predicted - target))

    return cscmae
