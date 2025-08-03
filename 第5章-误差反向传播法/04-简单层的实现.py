"""
实现神经网络中的基本计算层：乘法层和加法层
每个层包含前向传播(forward)和反向传播(backward)方法
用于构建计算图并自动求导
"""

# 乘法层的实现
class MulLayer:
    def __init__(self):
        # 初始化缓存变量，用于存储前向传播的输入值
        self.x = None  # 缓存第一个输入
        self.y = None  # 缓存第二个输入

    def forward(self, x, y):
        """
        前向传播：计算两个输入的乘积
        参数:
            x: 第一个输入值
            y: 第二个输入值
        返回:
            out: x * y 的结果
        """
        self.x = x  # 缓存输入值，用于反向传播
        self.y = y  # 缓存输入值，用于反向传播
        out = x * y  # 执行乘法运算
        return out

    def backward(self, dout):
        """
        反向传播：计算关于输入的梯度
        参数:
            dout: 上游传来的梯度
        返回:
            dx: 关于x的梯度
            dy: 关于y的梯度
        """
        dx = dout * self.y  # 应用链式法则计算x的梯度
        dy = dout * self.x  # 应用链式法则计算y的梯度
        return dx, dy


# ======== 测试乘法层 ======== #
apple = 100      # 苹果单价
apple_num = 2    # 苹果数量
tax = 1.1        # 税率

# 创建乘法层实例
mul_apple_layer = MulLayer()  # 计算苹果总价 (单价*数量)
mul_tax_layer = MulLayer()     # 计算含税总价 (总价*税率)

# 前向传播计算
apple_price = mul_apple_layer.forward(apple, apple_num)  # 100 * 2 = 200
price = mul_tax_layer.forward(apple_price, tax)           # 200 * 1.1 = 220

print(price)  # 输出最终价格: 220

# 反向传播计算梯度
dprice = 1  # 最终价格的梯度（初始值为1）
dapple_price, dtax = mul_tax_layer.backward(dprice)  # 计算苹果总价和税率的梯度
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # 计算苹果单价和数量的梯度

# 输出梯度值 (2.2, 110, 200)
print(dapple, dapple_num, dtax)


# 加法层的实现
class AddLayer:
    def __init__(self):
        """加法层不需要缓存输入值"""
        pass  # 无需初始化缓存变量

    def forward(self, x, y):
        """
        前向传播：计算两个输入的和
        参数:
            x: 第一个输入值
            y: 第二个输入值
        返回:
            out: x + y 的结果
        """
        out = x + y  # 执行加法运算
        return out

    def backward(self, dout):
        """
        反向传播：计算关于输入的梯度
        参数:
            dout: 上游传来的梯度
        返回:
            dx: 关于x的梯度 (等于dout)
            dy: 关于y的梯度 (等于dout)
        """
        dx = dout * 1  # 加法操作的梯度为1
        dy = dout * 1  # 加法操作的梯度为1
        return dx, dy


# ======== 测试组合计算（苹果+橘子） ======== #
apple = 100        # 苹果单价
apple_num = 2      # 苹果数量
orange = 150       # 橘子单价
orange_num = 3     # 橘子数量
tax = 1.1          # 税率

# 创建计算层实例
mul_apple_layer = MulLayer()      # 计算苹果总价
mul_orange_layer = MulLayer()      # 计算橘子总价
add_apple_orange_layer = AddLayer()  # 合并水果总价
mul_tax_layer = MulLayer()          # 计算含税总价

# 前向传播计算
apple_price = mul_apple_layer.forward(apple, apple_num)    # 100 * 2 = 200
orange_price = mul_orange_layer.forward(orange, orange_num)  # 150 * 3 = 450
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # 200 + 450 = 650
price = mul_tax_layer.forward(all_price, tax)              # 650 * 1.1 = 715

# 反向传播计算梯度
dprice = 1  # 最终价格的梯度（初始值为1）
dall_price, dtax = mul_tax_layer.backward(dprice)  # 计算总价和税率的梯度
# 注意：这里变量名dapple_price实际表示苹果总价的梯度，dorange_price表示橘子总价的梯度
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dapple_price)  # 计算橘子单价和数量的梯度
dapple, dapple_num = mul_apple_layer.backward(dapple_price)    # 计算苹果单价和数量的梯度

print(price)  # 输出最终价格: 715
# 输出各参数的梯度 (110, 2.2, 3.3, 165, 650)
print(dapple_num, dapple, dorange, dorange_num, dtax)