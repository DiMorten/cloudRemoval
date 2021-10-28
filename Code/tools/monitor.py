
import numpy as np
import keras.backend as K
import keras
from keras.callbacks import Callback
from time import time
import pdb
from tools.generator_amazon import generate_output_images
from icecream import ic
class Monitor(Callback):
    def __init__(self, validation):   
        super(Monitor, self).__init__()
        self.validation = validation 

#    def on_epoch_begin(self, epoch, logs={}):        
#        self.pred = []
#        self.targ = []

    def on_epoch_begin(self, batch, logs={}):
     
    #def on_batch_end(self, batch, logs={}):
        #print(batch)
        #if batch % 50 == 0:
        if True:
        
            batch_index = 0
            s2_cloudy = self.validation[batch_index][0][0]
            s2 = self.validation[batch_index][1][0]
            s1 = self.validation[batch_index][0][1]

            ic(s1.shape, s2_cloudy.shape, s2.shape)
            predicted = self.model.predict([s2_cloudy, 
                s1])
            ic(predicted.shape)
            ic(np.average(s1[0]))
            ic(np.max(s1[0]))
            
            ic(np.average(s2[0]))
            ic(np.average(s2_cloudy[0]))
            ic(np.average(predicted[0]))

            ims_name = 'sample_ims_monitor'
            ims_name = 'sample_ims_monitor_amazon'
            #  * 2000
            generate_output_images(s1[0]* 2000, s2[0] * 2000, s2_cloudy[0] * 2000, 
                np.squeeze(predicted)[0] * 2000,
                predicted_images_path = ims_name, scene_name = '2018')
            generate_output_images(s1[5]* 2000, s2[5] * 2000, s2_cloudy[5] * 2000, 
                np.squeeze(predicted)[5] * 2000,
                predicted_images_path = ims_name, scene_name = '2018_')

            #pdb.set_trace()
                

