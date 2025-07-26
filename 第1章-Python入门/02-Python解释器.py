# Python 基础语法示例

# ===== 1. 数据类型 =====
print(type(10))  # 输出: <class 'int'> - 整数类型
print(type(2.78))  # 输出: <class 'float'> - 浮点数类型
print(type("hello"))  # 输出: <class 'str'> - 字符串类型

# ===== 2. 变量 =====
x = 10  # 定义整型变量
print(x)  # 输出: 10

y = 3.14  # 定义浮点型变量
print(y * x)  # 输出: 31.4 - 变量相乘
print(type(x * y))  # 输出: <class 'float'> - 整数与浮点数相乘结果为浮点数

# ===== 3. 列表 =====
a = [1, 2, 3]  # 创建列表
print(a)  # 输出: [1, 2, 3]
print(len(a))  # 输出: 3 - 获取列表长度
print(a[0])  # 输出: 1 - 访问列表第一个元素(索引从0开始)

a[2] = 99  # 修改列表第三个元素
print(a)  # 输出: [1, 2, 99] - 修改后的列表

# 列表切片操作
print(a[0:2])  # 输出: [1, 2] - 索引0到1的元素(不包括索引2)
print(a[:2])  # 输出: [1, 2] - 同上，从开头到索引1的元素
print(a[:-1])  # 输出: [1, 2] - 从开头到倒数第二个元素
print(a[:-2])  # 输出: [1] - 从开头到倒数第三个元素

# ===== 4. 字典 =====
me = {'height': 180}  # 创建字典，存储键值对
print(me['height'])  # 输出: 180 - 访问键'height'对应的值

me['weight'] = 60  # 添加新的键值对
print(me)  # 输出: {'height': 180, 'weight': 60} - 更新后的字典

# ===== 5. 布尔类型 =====
hungry = True  # 布尔值True
sleepy = False  # 布尔值False
print(type(hungry))  # 输出: <class 'bool'> - 布尔类型
print(not hungry)  # 输出: False - 非运算
print(hungry and sleepy)  # 输出: False - 与运算
print(hungry or sleepy)  # 输出: True - 或运算

# ===== 6. if条件语句 =====
hungry = True
if hungry:
    print('I am hungry')  # 条件为真时执行，输出: I am hungry

hungry = False
if hungry:
    print('I am hungry')  # 条件为假，跳过
else:
    print('I am not hungry')  # 输出: I am not hungry
    print('I am sleepy')  # 输出: I am sleepy - else语句块中的第二条语句

# ===== 7. for循环语句 =====
for i in [1, 2, 3]:  # 遍历列表中的元素
    print(i)  # 依次输出: 1、2、3


# ===== 8. 函数 =====
# 定义无参数函数
def hello():
    print('hello world!')  # 函数体


hello()  # 调用函数，输出: hello world!


# 定义带参数函数
def hello(object):
    print(f'hello {object}!')  # 使用f-string格式化字符串


hello('cyb')  # 调用函数，输出: hello cyb!