
import os

Schedule = []                
'''0
Schedule.append("python main.py --phase train \
                                --dataset_name NRW \
                                --datasets_dir D:/Javier/Repo_Noa/SAR2Optical_Project/Datasets/ \
                                --date_mode t0")
'''
Schedule.append("python main.py --phase infer \
                                --predict E:/Jorge/cloudRemoval/checkpoint/DSen2-CR_001_noshadowNRW_60-0.4655.h5 \
                                --dataset_name NRW \
                                --datasets_dir D:/Javier/Repo_Noa/SAR2Optical_Project/Datasets/ \
                                --date_mode t0")


for i in range(len(Schedule)):
    os.system(Schedule[i])
