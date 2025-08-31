import math
import os

import numpy as np
import torch
from d2l import torch as d2l  #d2l:Dive into Deep Learning库

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#设置环境变量，避免运行PyTorch时可能出现的 OMP:Erroe 错误

n = 10000
a = torch.ones(n)   #a = [1,1,1,···,1] 长度为10000的全1向量
b = torch.ones(n)   #b同a
c = torch.zeros(n)  #c = [0,0,0,···,0]
timer = d2l.Timer() #创建一个计时对象，用来测量下面循环所用的时间
for i in range(n):
    c[i] = a[i] + b[i]
print(c)
print("{0:.5f} sec".format(timer.stop())) #测试出时间明显要长

timer.start()   #将时间重置为现在
d = a + b
print(d)
print("{0:.5f} sec".format(timer.stop()))
#时间会很小

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp((- 0.5 / sigma ** 2) * (x - mu) ** 2)
#求标准的正态分布概率密度函数(PDF)

## 可视化正态分布
x = np.arange(-7, 7, 0.01)  #生成从-7到7的数，步长0.01，会得到1400个点，作为横坐标
params = [(0, 1), (0, 2), (3, 1)]   #定义不同的参数分布
d2l.plot(   #d2l中封装好的绘图函数，相当于matplotlib
    x,
    [normal(x, mu, sigma) for mu, sigma in params],     #元组拆分
    xlabel='x',
    ylabel='p(x)',
    figsize=(4.5, 2.5),
    legend=[f'mean {mu}, std {sigma}' for mu, sigma in params]
)
d2l.plt.show()