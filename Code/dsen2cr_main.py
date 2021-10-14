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
import tifffile as tiff
import pathlib
ic.configureOutput(includeContext=True)

def run_dsen2cr(predict_file=None, resume_file=None):
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETUP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # TODO implement external hyperparam config file
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    model_name = 'DSen2-CR_001'  # model name for training

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
##    crop_size = 128  # crop size for training images
    if predict_file !=None:
        crop_size = 256
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

    input_shape = ((13, crop_size, crop_size), (2, crop_size, crop_size))

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

    if predict_file != None:

        test_filelist = entire_filelist.copy()
        train_filelist = test_filelist[:20]
        val_filelist = test_filelist[20:40]
        ic(test_filelist)
        
        
    else:
#        train_filelist = entire_filelist.copy()

#        ic(len(entire_filelist), int(len(entire_filelist)*0.1))
        train_filelist = entire_filelist[:-int(len(entire_filelist)*0.1)] # 2769
        val_filelist = entire_filelist[-int(len(entire_filelist)*0.1):]
        test_filelist = val_filelist.copy()

#        train_filelist = train_filelist[:32]
#        val_filelist = train_filelist[32:64]
#        ic(train_filelist)

    test_filelist = test_filelist[0:2]
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
        ic(predict_filelist)

        predict_dsen2cr(predict_file, model, predict_data_type, base_out_path, input_data_folder, predict_filelist,
                        batch_size, clip_min, clip_max, crop_size, input_shape, use_cloud_mask, cloud_threshold,
                        max_val_sar, scale)


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

