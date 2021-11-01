from __future__ import division

import argparse
import random

import keras.backend as K
import numpy as np
import tensorflow as tf
import tools.image_metrics as img_met
from dsen2cr_network import DSen2CR_model
from dsen2cr_tools import train_dsen2cr, predict_dsen2cr
from keras.optimizers import Nadam
from keras.utils import multi_gpu_model
from tools.dataIO import get_train_val_test_filelists
from icecream import ic
import pdb
K.set_image_data_format('channels_first')

import pickle
from predictAmazon import Image, ImageReconstruction
ic.configureOutput(includeContext=True)
from tools.image_metrics import metrics_get, metrics_get_mask
import cv2
import matplotlib.pyplot as plt 
import tifffile as tiff
from tools.dataIO import GeoReference_Raster_from_Source_data
import traceback
def run_dsen2cr(predict_file=None, resume_file=None):
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETUP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # TODO implement external hyperparam config file
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    remove_60m_bands = True

    model_name = 'DSen2-CR_001'  # model name for training
    if remove_60m_bands == True: 
        model_name = model_name + "_less60m"
        bands = 10
    else:
        bands = 13

    # model parameters
    num_layers = 16  # B value in paper
    feature_size = 256  # F value in paper

    # include the SAR layers as input to model
    include_sar_input = True

    # cloud mask parameters
    use_cloud_mask = True
    cloud_threshold = 0.2  # set threshold for binarisation of cloud mask

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup data processing param %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # input data preprocessing parameters
    scale = 2000
    max_val_sar = 2
    clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    clip_max = [[0, 0], [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]

    shuffle_train = True  # shuffle images at training time
    data_augmentation = True  # flip and rotate images randomly for data augmentation

    random_crop = True  # crop out a part of the input image randomly
    amazon_flag = True
##    crop_size = 128  # crop size for training images
    if predict_file !=None:
        crop_size = 256
        overlap = 0
        if amazon_flag == True:
            crop_size = 1024
            #crop_size = 256
            overlap = 0.1
    else:
        crop_size = 128  # crop size for training images

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    dataset_list_filepath = '../Data/datasetfilelist.csv'

    base_out_path = 'D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/results'
    input_data_folder = 'D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset'

    # training parameters
    initial_epoch = 0  # start at epoch number
    epochs_nr = 8  # train for this amount of epochs. Checkpoints will be generated at the end of each epoch
    if predict_file !=None:    
        batch_size = 1  # training batch size to distribute over GPUs
    else:
        batch_size = 8

    # define metric to be optimized
    loss = img_met.carl_error
    # define metrics to monitor
    metrics = [img_met.carl_error, img_met.cloud_mean_absolute_error,
               img_met.cloud_mean_squared_error, img_met.cloud_mean_sam, img_met.cloud_mean_absolute_error_clear,
               img_met.cloud_psnr,
               img_met.cloud_root_mean_squared_error, img_met.cloud_bandwise_root_mean_squared_error,
               img_met.cloud_mean_absolute_error_covered, img_met.cloud_ssim,
               img_met.cloud_mean_sam_covered, img_met.cloud_mean_sam_clear]

    # define learning rate
    lr = 7e-5

    # initialize optimizer
    optimizer = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Other setup parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if predict_file != None:
        predict_data_type = 'test'  # possible options: 'val' or 'test'
    else:
        predict_data_type = 'val'  # possible options: 'val' or 'test'

    log_step_freq = 1  # frequency of logging

    n_gpus = 1  # set number of GPUs
    # multiprocessing optimization setup
    use_multi_processing = True
    max_queue_size = 2 * n_gpus
    workers = 4 * n_gpus
    #workers = 1
    batch_per_gpu = int(batch_size / n_gpus)

    input_shape = ((bands, crop_size, crop_size), (2, crop_size, crop_size))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize session %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Configure Tensorflow session
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total % of the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3

    # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))

    # Set random seeds for repeatability
    random_seed_general = 42
    random.seed(random_seed_general)  # random package
    np.random.seed(random_seed_general)  # numpy package
    tf.set_random_seed(random_seed_general)  # tensorflow

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialize model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # single or no-gpu case
    if n_gpus <= 1:
        model, shape_n = DSen2CR_model(input_shape,
                                       batch_per_gpu=batch_per_gpu,
                                       num_layers=num_layers,
                                       feature_size=feature_size,
                                       use_cloud_mask=use_cloud_mask,
                                       include_sar_input=include_sar_input)
    else:
        # handle multiple gpus
        with tf.device('/cpu:0'):
            single_model, shape_n = DSen2CR_model(input_shape,
                                                  batch_per_gpu=batch_per_gpu,
                                                  num_layers=num_layers,
                                                  feature_size=feature_size,
                                                  use_cloud_mask=use_cloud_mask,
                                                  include_sar_input=include_sar_input)

        model = multi_gpu_model(single_model, gpus=n_gpus)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print('Model compiled successfully!')
    ic(model.summary())
    print("Getting file lists")
    # train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(dataset_list_filepath)
    # ic(train_filelist)
#    pdb.set_trace()
    # ic(test_filelist)
    # %%%%%%%%%%%%%%%%% override %%%%%%%%%
    
    with open("full_list.txt", "rb") as fp:   # Unpickling
        entire_filelist = pickle.load(fp)
    with open("val_list.txt", "rb") as fp:   # Unpickling
        val_filelist = pickle.load(fp)

    if predict_file != None:

        test_filelist = entire_filelist.copy()
        train_filelist = test_filelist[:20]
        #val_filelist = test_filelist[20:40]
        val_filelist = val_filelist[0:200]
        #ic(test_filelist)
        
        
    else:
#        train_filelist = entire_filelist.copy()

#        ic(len(entire_filelist), int(len(entire_filelist)*0.1))
        train_filelist = entire_filelist[:-int(len(entire_filelist)*0.1)] # 2769
        val_filelist = entire_filelist[-int(len(entire_filelist)*0.1):]
        test_filelist = val_filelist.copy()

#        train_filelist = train_filelist[:32]
#        val_filelist = train_filelist[32:64]
#        ic(train_filelist)


    print("Number of train files found: ", len(train_filelist))
    print("Number of validation files found: ", len(val_filelist))
    print("Number of test files found: ", len(test_filelist))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREDICT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if predict_file is not None:
        if predict_data_type == 'val':
            predict_filelist = val_filelist
        elif predict_data_type == 'test':
            predict_filelist = test_filelist
        else:
            raise ValueError('Prediction data type not recognized.')
        #ic(predict_filelist)

        print("Predicting using file: {}".format(predict_file))
        
        # load the model weights at checkpoint
        model.load_weights(predict_file)
        date = '2018'
        crop_sample_im = False
        im = Image(date = date, crop_sample_im = crop_sample_im)
        im.loadMask()

        ic(np.min(im.s1), np.min(im.s2), np.min(im.s2_cloudy))
        ic(np.average(im.s1), np.average(im.s2), np.average(im.s2_cloudy))
        ic(np.max(im.s1), np.max(im.s2), np.max(im.s2_cloudy))
        ic(np.std(im.s1), np.std(im.s2), np.std(im.s2_cloudy))

        ic(im.s1.dtype, im.s2.dtype, im.s2_cloudy.dtype)
        ic(im.mask.shape, im.s2.shape)
        # pdb.set_trace()
#        root_path = "D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/10m_all_bands/Sentinel2_2018"
        save_id = 'predictions_scratch'
        #save_id = 'predictions_pretrained'
        #save_id = 'predictions_remove60m'
        
        
        imReconstruction = ImageReconstruction(model, output_c_dim = bands, 
            patch_size=crop_size, overlap_percent = overlap)
        predictionLoad = False
        #pdb.set_trace()
        if predictionLoad == False:
            if remove_60m_bands == False:
                predictions = imReconstruction.infer(im.s2_cloudy, im.s1).astype(np.float32)
            else:
                predictions = imReconstruction.infer(im.s2_cloudy[np.r_[1:9,11:13]], im.s1).astype(np.float32)

            save_unnormalized = True
            # save_id = 'predictions_scratch'
            
            predictions = np.clip(predictions, 0, 10000.0)
                
            np.save(save_id+date+'.npy', predictions)
            if save_unnormalized == True:
                np.save(save_id+'_unnorm_' + date+'.npy', predictions*2000)
            #pdb.set_trace()

        else:
            predictions = np.load(save_id+date+'.npy').astype(np.float32)

        ic(np.average(im.s2), np.average(predictions), 
            np.std(im.s2), np.std(predictions))
        ic(np.min(predictions*2000), np.average(predictions*2000), 
            np.std(predictions*2000), np.max(predictions*2000))
            
        #pdb.set_trace()
        #===================================== Get metrics ======================#
        metrics_get_flag = True
        if metrics_get_flag == True:
            if remove_60m_bands == True:
                metrics_get(im.s2[np.r_[1:9,11:13]], predictions)
            else:
                metrics_get(im.s2, predictions)
                metrics_get_mask(im.s2[...,:-3], predictions[...,:-3], im.mask)

            #except Exception:
            #    print(traceback.format_exc())
            #    pass


            #pdb.set_trace()

        ic(np.average(im.s1), np.average(im.s2), np.average(im.s2_cloudy), np.average(predictions))
        ic(im.s1.dtype, im.s2.dtype, im.s2_cloudy.dtype)
        ic(im.s1.shape, im.s2.shape, im.s2_cloudy.shape)
        ic(predictions.dtype, predictions.shape)
        # pdb.set_trace()

        original_im_path = "D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset/10m_all_bands/Sentinel2_2018/COPERNICUS_S2_20180721_20180726_B1_B2_B3.tif"
        produced_im_path = save_id+"_"+date+".tif"
        GeoReference_Raster_from_Source_data(original_im_path, 
            predictions*2000, produced_im_path, bands = bands)
        GeoReference_Raster_from_Source_data(original_im_path, 
            im.s2*2000, "s2_"+date+".tif", bands = im.s2.shape[0])
        GeoReference_Raster_from_Source_data(original_im_path, 
            im.s2_cloudy*2000, "s2_cloudy_"+date+".tif", bands = im.s2_cloudy.shape[0])        
        pdb.set_trace()


        im.saveSampleIms(im.s1_unnormalized, im.s2, im.s2_cloudy, predictions)

        del im.s2_cloudy, im.s1
        #pdb.set_trace()



        #pdb.set_trace()
        ic(np.average(im.s2), np.average(predictions), 
            np.std(im.s2), np.std(predictions))
        
        #metrics_get(im.s2, predictions)
        im.s2 = im.s2*scale
        # predictions = predictions*scale

        ic(np.average(im.s2), np.average(predictions), 
            np.std(im.s2), np.std(predictions))
        

        #===================================== Save images TIF ======================#
        

        #s2_rgb = im.s2[1:4].astype(np.int)
        ic(im.s2.shape)
        #pdb.set_trace()
        im.s2 = np.transpose(im.s2, (1, 2, 0))
        ic(im.s2[:,:,1:4].astype(np.int16).shape)
        tiff.imsave('s2_saved_after.tif', im.s2[:,:,1:4].astype(np.int16), photometric='rgb')
        
        #plt.figure(figsize=(5,10))
        #plt.imshow(im.s2[:,:,1:4].astype(np.int16))
        #plt.axis('off')
        #plt.savefig('s2_rgb.png')
        #plt.show()
        #pdb.set_trace()        
        #predictions_rgb = predictions[1:4].astype(np.int)
        predictions = np.transpose(predictions, (1, 2, 0))
        ic(predictions[:,:,1:4].astype(np.int16).shape)

        ##tiff.imsave('predictions_saved.tif', predictions[:,:,1:4].astype(np.int16), photometric='rgb')
        
        #plt.figure(figsize=(5,10))
        #plt.imshow(predictions[:,:,1:4].astype(np.int16))
        #plt.axis('off')
        #plt.savefig('predictions_rgb.png')
        #plt.show()
        #==================== histogram
        plt.figure()
        n, bins, patches = plt.hist(predictions[:,:,1].flatten(), 300, density=True, facecolor='r',
            histtype = 'step')
        n, bins, patches = plt.hist(predictions[:,:,2].flatten(), 300, density=True, facecolor='g',
            histtype = 'step')
        n, bins, patches = plt.hist(predictions[:,:,3].flatten(), 300, density=True, facecolor='b',
            histtype = 'step')

        #plt.show()
        #plt.figure()
        n, bins, patches = plt.hist(im.s2[:,:,1].flatten(), 300, density=True, facecolor='r',
            histtype = 'step')
        n, bins, patches = plt.hist(im.s2[:,:,2].flatten(), 300, density=True, facecolor='g',
            histtype = 'step')
        n, bins, patches = plt.hist(im.s2[:,:,3].flatten(), 300, density=True, facecolor='b',
            histtype = 'step')
        #plt.xlim(500, 1800)
        plt.legend(['Predictions','Predictions','Predictions', 'Target','Target','Target'])
        plt.show()

        

    else:
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        train_dsen2cr(model, model_name, base_out_path, resume_file, train_filelist, val_filelist, lr, log_step_freq,
                      shuffle_train, data_augmentation, random_crop, batch_size, scale, clip_max, clip_min, max_val_sar,
                      use_cloud_mask, cloud_threshold, crop_size, epochs_nr, initial_epoch, input_data_folder,
                      input_shape, max_queue_size, use_multi_processing, workers)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSen2-CR model code')
    parser.add_argument('--predict', action='store', dest='predict_file', help='Predict from model checkpoint.')
    parser.add_argument('--resume', action='store', dest='resume_file', help='Resume training from model checkpoint.')
    args = parser.parse_args()

    run_dsen2cr(args.predict_file, args.resume_file)

    
    

