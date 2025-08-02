# coding: utf-8
import sys, os

# 设置系统路径以导入父目录中的文件
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt

# 从自定义模块导入数据加载器和神经网络类
from 数据.mnist import load_mnist  # MNIST数据加载器
from 共用资源.two_layer_net import TwoLayerNet  # 两层神经网络实现

# 1. 数据加载与预处理
# 加载MNIST数据集，并进行归一化和one-hot编码处理
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 2. 神经网络初始化
# 创建输入层784个神经元(28x28像素)、隐藏层50个神经元、输出层10个神经元(10个数字类别)的神经网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 3. 训练参数设置
iters_num = 10000  # 总迭代次数
train_size = x_train.shape[0]  # 训练集大小(60000)
batch_size = 100  # 批处理大小
learning_rate = 0.1  # 学习率

# 4. 初始化记录变量
train_loss_list = []  # 记录每个批次的训练损失
train_acc_list = []  # 记录每个epoch的训练准确率
test_acc_list = []  # 记录每个epoch的测试准确率

# 计算每个epoch包含的迭代次数(60000/100=600)
iter_per_epoch = max(train_size / batch_size, 1)

# 5. 训练循环
for i in range(iters_num):
    # 随机选择批量数据
    batch_mask = np.random.choice(train_size, batch_size)  # 从训练集中随机选择batch_size个索引
    x_batch = x_train[batch_mask]  # 批量输入数据
    t_batch = t_train[batch_mask]  # 批量标签数据

    # 计算梯度 - 使用反向传播替代数值梯度计算以提高效率
    # grad = network.numerical_gradient(x_batch, t_batch)  # 数值梯度方法(较慢)
    grad = network.gradient(x_batch, t_batch)  # 反向传播方法(更高效)

    # 更新参数（权重和偏置）
    for key in ('W1', 'b1', 'W2', 'b2'):  # 遍历所有参数
        network.params[key] -= learning_rate * grad[key]  # 梯度下降更新

    # 记录当前批次的损失
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 6. 评估与记录
    # 每完成一个epoch(约600次迭代)，评估模型性能
    if i % iter_per_epoch == 0:
        # 计算整个训练集和测试集的准确率
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)

        # 记录准确率
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # 输出当前性能
        print(f"epoch: {int(i / iter_per_epoch)}, train acc: {train_acc:.4f}, test acc: {test_acc:.4f}")

# 7. 结果可视化
# 绘制训练和测试准确率曲线
plt.figure(figsize=(10, 6))
x = np.arange(len(train_acc_list))  # 创建epoch数轴
plt.plot(x, train_acc_list, label='train acc', marker='o')  # 训练准确率曲线
plt.plot(x, test_acc_list, label='test acc', linestyle='--', marker='s')  # 测试准确率曲线
plt.xlabel("Epochs")  # x轴标签
plt.ylabel("Accuracy")  # y轴标签
plt.ylim(0, 1.0)  # 设置y轴范围(0-100%)
plt.legend(loc='lower right')  # 图例位置
plt.title('Training and Test Accuracy over Epochs')  # 图表标题
plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
plt.show()  # 显示图表