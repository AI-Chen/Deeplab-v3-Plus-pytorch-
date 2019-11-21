import torch
import torchvision.utils as vutils
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from utils.data_utils import Mydataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime
from deeplab_v3p import DeepLabv3_plus
import tensorflow as tf
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.logging.set_verbosity(tf.logging.INFO)

class args:
    train_path = 'dataset/train_path_list.csv'
    val_path = 'dataset/val_path_list.csv'
    result_dir = 'Result/'
    batch_size = 2
    learning_rate = 0.001
    max_epoch = 50

best_train_acc = 0.80

now_time = datetime.now()
time_str = datetime.strftime(now_time,'%m-%d_%H-%M-%S')

log_dir = os.path.join(args.result_dir,time_str)
if not os.path.exists(log_dir):
     os.makedirs(log_dir)

writer = SummaryWriter(log_dir)
#---------------------------1、加载数据---------------------------
#数据预处理设置
normMean = [0.46099097, 0.32533738, 0.32106236]
normStd = [0.20980413, 0.1538582, 0.1491854]
normTransfrom = transforms.Normalize(normMean, normStd)
transform = transforms.Compose([
        transforms.ToTensor(),
        normTransfrom,
    ])
#构建Mydataset实例
train_data = Mydataset(path=args.train_path,transform=transform)
val_data = Mydataset(path=args.val_path,transform=transform)
#构建DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

#---------------------------2、定义网络---------------------------
net = DeepLabv3_plus(nInputChannels=3, n_classes=6, os=8, pretrained=True, _print=True)
net.cuda()

#---------------------------3、初始化预训练权重、定义损失函数、优化器、设置超参数、---------------------------
if torch.cuda.is_available():
    w = torch.Tensor([0.71280016, 0.77837713, 0.93428148, 1.0756635, 16.18921045, 28.26338505]).cuda()
else:
    w = torch.Tensor([0.71280016, 0.77837713, 0.93428148, 1.0756635, 16.18921045, 28.26338505])
criterion = nn.CrossEntropyLoss(weight= w).cuda()   #选择损失函数
optimizer = optim.SGD(net.parameters(),lr=args.learning_rate,momentum=0.9,dampening=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=10,gamma=0.5)

#---------------------------4、训练网络---------------------------
for epoch in range(args.max_epoch):
    loss_sigma = 0.0
    acc_sigma = 0.0
    loss_val_sigma = 0.0
    acc_val_sigma = 0.0
    net.train()
    for i,data in enumerate(train_loader):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        labels = labels.long().cuda()
        optimizer.zero_grad()
        outputs = net.forward(inputs)

        loss = criterion(outputs, labels)
        predicts = torch.argmax(outputs, dim=1)
        acc_train = accuracy_score(np.reshape(labels.cpu(),[-1]),np.reshape(predicts.cpu(),[-1]))
        loss.backward()
        optimizer.step()
        # 统计预测信息
        loss_sigma += loss.item()
        acc_sigma += acc_train
        if i % 10 == 9 :
            loss_avg = loss_sigma /10
            acc_avg = acc_sigma/10
            loss_sigma = 0.0
            acc_sigma = 0.0
            tf.logging.info("Training:Epoch[{:0>3}/{:0>3}] Iter[{:0>3}/{:0>3}] Loss:{:.4f} Acc:{:.4f}".format(
                epoch + 1, args.max_epoch,i+1,len(train_loader),loss_avg,acc_avg))
            writer.add_scalar("LOSS", loss_avg, epoch)
            writer.add_scalar("LEARNING_RATE", scheduler.get_lr()[0], epoch)
            # 保存模型
            if (acc_avg) > best_train_acc:
                # 保存acc最高的模型
                net_save_path = os.path.join(log_dir, 'net_params.pkl')
                torch.save(net.state_dict(), net_save_path)
                best_train_acc = acc_avg
                tf.logging.info('Save model successfully to "%s"!' % (log_dir + 'net_params.pkl'))

    net.eval()
    for i, data in enumerate(val_loader):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        labels = labels.long().cuda()
        with torch.no_grad():
            outputs = net.forward(inputs)
        predicts = torch.argmax(outputs, dim=1)
        acc_val = accuracy_score(np.reshape(labels.cpu(), [-1]), np.reshape(predicts.cpu(), [-1]))
        loss_val = criterion(outputs, labels)
        # 统计预测信息
        loss_val_sigma += loss_val.item()
        #acc_val_sigma += acc_val
    tf.logging.info("After 1 epoch：acc_val:{:.4f},loss_val:{:.4f}".format(acc_val_sigma/(len(val_loader)), loss_val_sigma/(len(val_loader))))
    acc_val_sigma = 0.0
    loss_val_sigma = 0.0

writer.close()
net_save_path = os.path.join(log_dir,'net_params_end.pkl')
torch.save(net.state_dict(),net_save_path)