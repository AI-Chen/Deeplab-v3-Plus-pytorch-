# Deeplab v3-Plus
Deeplab v3-plus for semantic segmentation of remote sensing（pytorch）

## 数据集：
在ISPRS Vaihigen 2D语义标签比赛数据集上评估了deeplab v3+的表现。该数据集由33张大小不同的高分辨率遥感影像组成，每张影像都是从德国Vaihigen市高空中获取的真正射影象（TOP）。在某种程度上，这个数据集的遥感印象与普通的自然影像没有差别，他们均是由三个通道组成。所以，我们可以将其看作是普通图片。数据集还包括与每个影像对应的归一化数字表面模型（nDSM）。在这33张图像中，16张提供了精细标注的ground truth，其余17张并未公布ground truth。整个数据集包含六个类别：不透水表面、建筑、低植被、树木、汽车、杂波/背景。

## 文件夹列表：
+ dataset：存放处理好的数据集的文件夹（处理好的数据集太大没办法放上来，所以以下train、val、test文件夹是没有上传的。可以去官网http://www2.isprs.org/commissions/comm2/wg4/vaihingen-2d-semantic-labeling-contest.html
 下载然后自己处理，裁剪成512×512的图片，并自己划分训练集与验证集）
  + train文件夹:存放训练集图片；
  + val文件夹:存放验证集图片
  + test文件夹:存放测试集图片
  + train_path_list.csv:存放训练集图片及标签的实际存储位置的索引文件
  + val_path_list.csv:存放验证集图片及标签的实际存储位置的索引文件
  + test_path_list.csv:存放测试集图片实际存储位置的索引文件
+ Result：保存模型参数与预测结果的文件夹
+ utils：
  + data_utils:读取数据
  + color_utils:给预测图上色
  + Median_frequency_balance.py：计算各类别权重
  + compute_mean.py：计算数据集均值、方差
+ deeplab_v3p.py:deeplab v3 plus的模型
+ deeplab_v3p_train.py:想要训练模型直接运行这个文件
+ predict.py:多尺度测试。使用训练好的模型进行预测

## 预测结果展示：
见Result/deeplab v3P/1.png与Result/deeplab v3P/2.png，这个文件夹下其他两张图是这两张预测图对应的原图
