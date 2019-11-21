import numpy as np
import cv2

# 给标签图上色

def color_annotation(img, output_path):

    '''

    给class图上色

    '''

    color = np.ones([img.shape[0], img.shape[1], 3])

    color[img==0] = [255, 255, 255] #不透水地表

    color[img==1] = [255, 0, 0]     #建筑

    color[img==2] = [0, 255, 0]       # 树木

    color[img==3] = [255, 255, 0] # 低层植被

    color[img==4] = [0, 255, 255]   # 汽车

    color[img == 5] = [0, 0, 255]  #水面



    cv2.imwrite(output_path,color)