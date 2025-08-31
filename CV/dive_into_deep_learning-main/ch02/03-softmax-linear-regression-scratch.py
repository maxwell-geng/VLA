import sys
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from d2l import torch as d2l
#import d2lutil.common as common

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

## 读取小批量数据
batch_size = 256
trans = transforms.ToTensor()       #将图片转换为tensor
mnist_train  = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True) #下载并加载训练数据集
mnist_test  = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True) #下载并加载测试数据集
print(len(mnist_train))  # 输出训练集样本个数
print('11111111')

#注意到，我们的batch_size是256，这意味着在一个epoch中，我们会训练235次

## 展示部分数据
def get_fashion_mnist_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
    #以列表形式返回标签的文本描述

def show_fashion_mnist(images, labels):
    d2l.use_svg_display()   #设置使用SVG格式显示图像
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))    #用于创建一个包含多个子图的Matplotlib图形对象
    #表示生成1行len(images)列的子图，figsize=(12, 12)设置整个图像的尺寸。_接收整个图形对象（未使用）
    for f, img, lbl in zip(figs, images, labels): #zip函数将多个可迭代的对象打包成一个元组列表
        # 分别用f表示当前的图像对象，img表示当前的图像数据，lbl表示当前的标签
        # 这里的img是一个tensor，形状为(1, 28, 28)，需要将其转换为numpy数组
        # view(28, 28)
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)        # 设置标题为标签
        f.axes.get_xaxis().set_visible(False)   # 不显示x轴
        f.axes.get_yaxis().set_visible(False)   # 不显示y轴
    plt.show() #用于显示所有已创建的Matplotlib图形窗口


train_data, train_targets = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# data.DataLoader是PyTorch中用于加载数据的工具，它可以将数据集分成小批量（batch）进行训练
# iter()把数据集转换为迭代器，next()获取迭代器的下一个元素
#展示部分训练数据
show_fashion_mnist(train_data[0:10], get_fashion_mnist_labels(train_targets[0:10]))

# 初始化模型参数
num_inputs = 784    #28 * 28 # 每张图片的像素数
num_outputs = 10    # 10个输出类别，分别对应10种服装类别

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 从正态分布中随机生成权重矩阵W，均值为0，标准差为0.01
b = torch.zeros(num_outputs, requires_grad=True)
# 生成偏置向量b，初始值为0
v_W = torch.zeros_like(W)  # 初始化动量v为0，形状与W相同
v_b = torch.zeros_like(b)  # 初始化偏置的动量v_b为0，形状与b相同
v_params = [v_W, v_b]  # 将动量参数放入列表中，便于后续使用

# 定义模型
def softmax(X):
    X_exp = X.exp()     # X(256,10) 对每个元素取指数
    partition = X_exp.sum(dim=1, keepdim=True)  #对每个样本的得分进行求和，keepdim=True保持维度(256, 1)
    return X_exp / partition  # 这里应用了广播机制

def sgd(params, lr, batch_size):
    with torch.no_grad():  # with torch.no_grad() 则主要是用于停止autograd模块的工作，
        for param in params:
            param -= lr * param.grad / batch_size  ##  这里用param = param - lr * param.grad / batch_size会导致导数丢失， zero_()函数报错
            param.grad.zero_()  ## 导数如果丢失了，会报错‘NoneType’ object has no attribute ‘zero_’

def sgd_momentum(params, vs, lr, batch_size, beta=0.9):
    with torch.no_grad():
        for param, v in zip(params, vs):
            v[:] = beta * v + lr * param.grad / batch_size
            param -= v
            param.grad.zero_()

def net(X):
    # 输入的 X 通常为(256(batch_size), 1, 28, 28), 即一批28*28的灰度图
    # 现在将其reshape为 X(256, 784), -1让PyTorch自动计算batch_size
    return softmax(torch.matmul(X.reshape(-1, num_inputs), W) + b)


# 定义损失函数
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 计算分类准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()
# .mean()取平均，就是准确率
# .item()将张量标量转换为Python数字

# 计算这个训练集的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    # y.shape[0]是每个batch的样本数，可能每个batch样本数量不一样，所以用总数除以总样本个数
    return acc_sum / n


num_epochs, lr = 25, 0.1
#定义epoch和学习率


# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer_fn=None, vs=None, beta=0.9):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.zero_()

            l.backward()

            # 使用优化器函数
            if optimizer_fn is not None:
                if optimizer_fn.__name__ == "sgd_momentum":
                    optimizer_fn(params, vs, lr, batch_size, beta)
                else:
                    optimizer_fn(params, lr, batch_size)
            else:
                sgd(params, lr, batch_size)

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


# 训练模型
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
          [W, b], lr, optimizer_fn=sgd)


# 预测模型
for X, y in test_iter:
    break
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])
