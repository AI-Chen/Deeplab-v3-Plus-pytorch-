from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import  numpy as np
import torchvision.transforms as transforms

class Mydataset(Dataset):
    def __init__(self, path, transform=None, target_transform=None):
        '''
        :param path: 存储有图片存放地址、对应标签的文件的地址；
        :param transform: 定义了各种包括随即裁剪、旋转、仿射等在内的对图像的预处理操作
        :param target_transform:
        '''
        data = pd.read_csv(path)  # 获取csv表中的数据
        imgs = []
        for i in range(len(data)):
            imgs.append((data.ix[i,1], data.ix[i,2]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = Image.open(fn).convert('RGB')
        gt = np.array(Image.open(label))
        # 进行数据增强，并转换数据格式为tensor
        if self.transform is not None:
            img = self.transform(img)
        #gt = gt.astype(np.long)
        return img, gt

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    # 数据预处理设置
    normMean = [0.46099097, 0.32533738, 0.32106236]
    normStd = [0.20980413, 0.1538582, 0.1491854]
    normTransfrom = transforms.Normalize(normMean, normStd)
    transform = transforms.Compose([
        transforms.ToTensor(),
        normTransfrom,
    ])
    # 构建Mydataset实例
    train_data = Mydataset(path='../dataset/train_path_list.csv', transform=transform)
    img, gt = train_data.__getitem__(0)
    print(img.shape,gt)