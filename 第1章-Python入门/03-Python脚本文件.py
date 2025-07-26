"""
class 类名：
    def __init__(self, 参数, …): # 构造函数
       ...
    def 方法名1(self, 参数, …): # 方法1
       ...
    def 方法名2(self, 参数, …): # 方法2
       ...

__init__方法，这是进行初始化的方法，也称为构造函数（constructor）,只在生成类的实例时被调用一次。
此外，在方法的第一个参数中明确地写入表示自身（自身的实例）的self是Python的一个特点
"""

# 定义一个名为 Man 的类
class Man():
    """
    人类类，表示一个人的基本信息和行为
    """

    def __init__(self, name):
        """
        构造函数(初始化方法)
        当创建类的新实例时自动调用

        参数:
        self: 类的实例自身(约定名称)
        name: 人的名字
        """
        self.name = name  # 将传入的name参数绑定到实例的name属性
        print(f"Initialized! Name: {name}")  # 创建实例时打印初始化信息

    def hello(self):
        """
        问候方法
        """
        print(f"Hello, {self.name}!")  # 使用实例的name属性进行问候

    def goodbye(self):
        """
        告别方法
        """
        print(f"Goodbye, {self.name}!")  # 使用实例的name属性进行告别


# 创建Man类的实例
m = Man("CYB")  # 实例化过程，调用__init__方法，传递"CYB"作为name参数

# 调用实例方法
m.hello()  # 输出: Hello, CYB!
m.goodbye()  # 输出: Goodbye, CYB!