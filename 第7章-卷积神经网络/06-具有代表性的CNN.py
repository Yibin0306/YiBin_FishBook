# LeNet
"""
LeNet在1998年被提出，是进行手写数字识别的网络。
和“现在的CNN”相比，LeNet有几个不同点。第一个不同点在于激活函数。
LeNet中使用sigmoid函数，而现在的CNN中主要使用ReLU函数。
此外，原始的LeNet中使用子采样（subsampling）缩小中间数据的大小，而现在的CNN中Max池化是主流。
"""

# AlexNet
"""
AlexNet叠有多个卷积层和池化层，最后经由全连接层输出结果。
虽然结构上AlexNet和LeNet没有大的不同，但有以下几点差异。
• 激活函数使用ReLU。
• 使用进行局部正规化的LRN（Local Response Normalization）层。
• 使用Dropout
"""