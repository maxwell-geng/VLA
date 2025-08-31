import torch

print('1.自动梯度计算')
x = torch.arange(4.0, requires_grad=True)
# torch.arange生成一个1维张量，内容是由0.0到4.0的连续整数；
# 后面表示这个张量是需要梯度的，PyTorch在后续的运算中会为其构建计算图，便于计算梯度

print('x:', x)
print('x.grad:', x.grad)

y = 2 * torch.dot(x, x)  # 2.记录目标值的计算
#y为向量x点积向量x，PyTorch会记录下这一步运算，构建计算图
print('y:', y)
y.backward()  # 3.执行它的反向传播函数
# 调用反向传播，自动计算y对x的梯度，并存到x.grad中
print('x.grad:', x.grad)  # 4.访问得到的梯度
print('x.grad == 4*x:', x.grad == 4 * x)

## 计算另一个函数
x.grad.zero_()
#清零x.grad中的梯度，在PyTorch里，梯度是累加的，而不是每次计算时自动覆盖
y = x.sum()
print('y:', y)
y.backward()
print('x.grad:', x.grad)
y.backward()    #若不清零
print('##x.grad:', x.grad)

# 非标量变量的反向传播
x.grad.zero_()
print('x:', x)
y = x * x
#y是x逐项相乘，特别的，如果两个向量维数不一样时，PyTorch会尝试广播机制处理
#从后往前对齐两个张量的形状；如果维度不相等，但其中一个是1，就把它复制扩展；如果维度不匹配又不是1，就报错。
print("y是什么东西：",y)
y.sum().backward()
print('x.grad:', x.grad)


def f(a):
    b = a * 2
    print(b.norm())         # norm默认为求L2范数
    while b.norm() < 1000:  # 求L2范数：元素平方和的平方根
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


print('2.Python控制流的梯度计算')
a = torch.tensor(-2.0)  # 初始化变量,此时a为一个标量
a.requires_grad_(True)  # 将梯度赋给想要对其求偏导数的变量
# 让PyTorch跟踪这个变量的运算，后面能自动求导
print('a:', a)
d = f(a)  # 记录目标函数
print('d:', d)
d.backward()  # 3.执行目标函数的反向传播函数
print('a.grad:', a.grad)  # 4.获取梯度