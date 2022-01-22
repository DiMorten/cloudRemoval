
import os

Schedule = []                

Schedule.append("python main.py --phase train \
                                --dataset_name NRW \
                                --datasets_dir D:/Jorge/dataset/ \
                                --date_mode t0")


for i in range(len(Schedule)):
    os.system(Schedule[i])
