"""
使用神经网络解决问题时，也需要首先使用训练数据（学习数据）进行权重参数的学习；
进行推理时，使用刚才学习到的参数，对输入数据进行分类。

先实现神经网络的“推理处理”。这个推理处理也称为神经网络的前向传播（forward propagation）
"""

# MNIST数据集
"""
MNIST数据集是由0到9的数字图像构成的。训练图像有6万张，测试图像有1万张，这些图像可以用于学习和推理。
MNIST数据集一般先用训练图像进行学习，再用学习到的模型度量能在多大程度上对测试图像进行正确的分类。
"""

from torchvision import datasets
import sys, os
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
from 数据.mnist import load_mnist

# 第一次调用会花费几分钟……
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# 输出各个数据的形状
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)
print(x_test.shape)  # (10000, 784)
print(t_test.shape)  # (10000,)