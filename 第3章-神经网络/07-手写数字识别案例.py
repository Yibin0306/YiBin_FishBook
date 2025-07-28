"""
由于书本的MNIST数据集链接失效，以下为其它方式训练数据集
可以学完书本回头再看这个案例
"""

import os
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.nn import Conv2d, Linear, ReLU
from torch.nn import MaxPool2d
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Dataset:创建数据集的函数；__init__:初始化数据内容和标签
# __geyitem:获取数据内容和标签
# __len__:获取数据集大小
# dataloader:数据加载类，接受来自dataset已经加载好的数据集
# torchbision:图形库，包含预训练模型，加载数据的函数、图片变换，裁剪、旋转等
# torchtext:处理文本的工具包，将不同类型的额文件转换为datasets

# 预处理：将两个步骤整合在一起
transform = transforms.Compose({
    transforms.ToTensor(),  # 将灰度图片像素值（0~255）转为Tensor（0~1），方便后续处理
    # transforms.Normalize((0.1307,),(0.3081)),    # 归一化，均值0，方差1;mean:各通道的均值std：各通道的标准差inplace：是否原地操作
})

# 加载数据集
# 训练数据集
train_data = MNIST(root='../数据', train=True, download=True, transform=transform)
train_loder = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# transform：指示加载的数据集应用的数据预处理的规则，shuffle：洗牌，是否打乱输入数据顺序
# 测试数据集
test_data = MNIST(root='../数据', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度：{}".format(train_data_size))
print("测试数据集的长度：{}".format(test_data_size))

# 构建模型
# 模型主要由两个卷积层，两个池化层，以及三个全连接层构成，激活函数使用relu.
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = MaxPool2d(2)
        self.linear1 = Linear(320, 128)
        self.linear2 = Linear(128, 64)
        self.linear3 = Linear(64, 10)
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

# 损失函数CrossentropyLoss
model = MnistModel() # 实列化
criterion = nn.CrossEntropyLoss() # 交叉熵损失，相当于Softmax+Log+NllLoss
# 线性多分类模型Softmax,给出最终预测值对于10个类别出现的概率，Log:将乘法转换为加法，减少计算量，保证函数的单调性
# NLLLoss:计算损失，此过程不需要手动one-hot编码，NLLLoss会自动完成

# SGD，优化器，梯度下降算法e
optimizer = torch.optim.SGD(model.parameters(), lr=0.14) # lr学习率

# 模型训练
def train():
    # index = 0
    for index , data in enumerate(train_loder): # 获得训练数据以及对应标签
        # for data in train_loader
        input, target = data # input为输入数据，target为标签
        y_predict = model(input) # 模型预测
        loss = criterion(y_predict, target)
        optimizer.zero_grad() # 梯度清零
        loss.backward() # loss值反向传播
        optimizer.step() # 更新参数
        # index += 1
        if index % 100 == 0: # 每一百次保存一次模型，打印损失
            torch.save(model.state_dict(), '../数据/Model/model.pkl') # 保存模型
            torch.save(optimizer.state_dict(), '../数据/Model/optimizer.pkl')
            print("训练次数为：{}，损失值为：{}".format(index, loss.item()))

# 加载模型
if os.path.exists('../数据/Model/model.pkl'):
    model.load_state_dict(torch.load('../数据/Model/model.pkl')) # 加载保存模型的参数

# 模型测试
def test():
    correct = 0 # 正确预测的个数
    total = 0  # 总数
    with torch.no_grad(): # 测试不用计算梯度
        for data in test_loader:
            inputs, target = data
            output = model(inputs) # output输出10个预测取值，概率最大的为预测数
            probability, predict = torch.max(input=output.data, dim=1)  # 返回一个元祖，第一个为最大概率值，第二个为最大概率值的下标
            # loss = criterion(output, target)
            total += target.size(0)  # target是形状为（batch_size,1)的矩阵，使用size（0）取出该批的大小
            correct += (predict == target).sum().item()  # predict 和target均为（batch_size,1)的矩阵，sum求出相等的个数
        print("测试准确率为：%.6f" % (correct / total))

#测试识别函数
if __name__ == '__main__':
    #训练与测试
    for i in range(15):#训练和测试进行5轮
        print({"————————第{}轮测试开始——————".format (i + 1)})
        train()
        test()


def test_mydata():
    image = Image.open('../数据/img_1.png')   #读取自定义手写图片
    image = image.resize((28, 28))   # 裁剪尺寸为28*28
    image = image.convert('L')  # 转换为灰度图像
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.resize(1, 1, 28, 28)
    output = model(image)
    probability, predict = torch.max(output.data, dim=1)
    print("此手写图片值为：%d,其最大概率为：%.2f " % (predict[0], probability))
    plt.title("此手写图片值为：{}".format((int(predict))), fontname='SimHei')
    plt.imshow(image.squeeze())
    plt.show()

# 测试主函数
if __name__ == '__main__':
    test_mydata()