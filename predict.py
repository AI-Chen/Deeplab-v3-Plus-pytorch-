from utils.data_utils import Mydataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from deeplab_v3p import DeepLabv3_plus
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from  utils.color_utils import color_annotation
from sklearn.metrics import accuracy_score
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

pretrained_path = 'Result/11-18_12-15-21/net_params.pkl'
output_path = 'Result/11-18_12-15-21/'
scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
OA_all = []
classes = 6
normMean = [0.46099097, 0.32533738, 0.32106236]
normStd = [0.20980413, 0.1538582, 0.1491854]
crop_h = 512
crop_w = 512

# 构建模型，载入训练好的权重参数
net = DeepLabv3_plus(nInputChannels=3, n_classes=6, os=8, pretrained=True, _print=True)
net.eval()
if torch.cuda.is_available():
    #支持cuda计算的情况下
    net = net.cuda()
    pretrained_dict = torch.load(pretrained_path,map_location=torch.device('cuda'))
    net.load_state_dict(pretrained_dict)
else:
    # 不支持cuda计算的情况下
    pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    net.load_state_dict(pretrained_dict)

# 数据预处理设置
normMean = [0.46099097, 0.32533738, 0.32106236]
normTransfrom = transforms.Normalize(normMean, normStd)
transform = transforms.Compose([
    transforms.ToTensor(),
    normTransfrom,
])
# 构建Mydataset实例
test_data = Mydataset(path='dataset/test_path_list.csv', transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

def net_process(model, image, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if torch.cuda.is_available():
        input = input.unsqueeze(0).cuda()
    else:
        input = input.unsqueeze(0)
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model,image_crop)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction

if __name__ == '__main__':
    for i,data in enumerate(test_loader):
        print(i)
        input, label = data
        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            base_size = 0
            if h > w:
                base_size = h
            else:
                base_size = w
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size / float(h) * w)
            else:
                new_h = round(long_size / float(w) * h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(net, image_scale, classes, crop_h, crop_w, h, w, normMean)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)
        color_annotation(prediction, output_path + str(i) + ".png")
        OA = accuracy_score(np.reshape(prediction,[-1]), np.reshape(label,[-1]))
        print("the " + str(i) + "th image's OA:" + str(OA))
        OA_all.append(OA)
    print(np.mean(OA_all))









