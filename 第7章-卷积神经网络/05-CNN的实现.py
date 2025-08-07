"""
网络的构成是“Convolution - ReLU - Pooling -Affine - ReLU - Affine - Softmax”
我们将它实现为名为SimpleConvNet的类。
SimpleConvNet的初始化（__init__），取下面这些参数
参数
• input_dim―输入数据的维度：（通道，高，长）
• conv_param―卷积层的超参数（字典）。字典的关键字如下：
    filter_num―滤波器的数量
    filter_size―滤波器的大小
    stride―步幅
    pad―填充
• hidden_size―隐藏层（全连接）的神经元数量
• output_size―输出层（全连接）的神经元数量
• weight_int_std―初始化时权重的标准差
"""

# 源码
import 共用资源.simple_convnet