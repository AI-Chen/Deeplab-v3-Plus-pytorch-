# coding: utf-8

import numpy as np
import cv2
import random
import pandas as pd

"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""


train_csv_path = '../dataset/train_path_list.csv'

CNum = 960     # 挑选多少图片进行计算

img_h, img_w = 512, 512
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

data = pd.read_csv(train_csv_path)  #获取csv表中的数据
data = data.sample(frac=0.20)   # shuffle , 随机挑选图片
data = data.reset_index(drop = True) #重新添加index，drop表示丢弃原有index一列

for i in range(CNum):
    img_path = '../' + data.ix[i,1]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_h, img_w))
    img = img[:, :, :, np.newaxis]
    imgs = np.concatenate((imgs, img), axis=3)
    print(i)

imgs = imgs.astype(np.float32)/255.


for i in range(3):
    pixels = imgs[:,:,i,:].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse() # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

