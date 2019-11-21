from __future__ import print_function
import os, sys
import numpy as np
import cv2


def calcalate_weight(count):
    frequence = np.array(count) / sum(count)

    sort_frequence = sorted(frequence)
    size = len(frequence)

    if size % 2 == 0:
        median = (sort_frequence[size//2] + sort_frequence[size//2-1])/2
    if size % 2 == 1:
        median = sort_frequence[(size - 1) // 2]

    class_weight = median / frequence

    return class_weight


class_pixels_count = [21815349, 20417332, 16272917, 18110438, 945687, 526083]
train_dict = {0: 20575010,  1: 18841600,  2: 15697486,  3: 13634255,  4: 905904,  5:   518900,
              }
train_count = list(train_dict.values())

val_dict = {0: 1240339,  1: 1575732,  2: 2412952,  3: 2638662,  4: 39783,  5:   7183,
              }
val_count = list(val_dict.values())

train_weight = calcalate_weight(train_count)
print(train_weight)
#[ 0.71280016  0.77837713  0.93428148  1.0756635  16.18921045 28.26338505]

val_weight = calcalate_weight(val_count)
print(val_weight)
# [  1.13520215   0.89357549   0.58353233   0.53361723  35.39289395   196.02331895]
