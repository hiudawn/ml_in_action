import random
import numpy as np
import matplotlib.pyplot as plt


def load_dataset():
    """
    前两行两个值分别是X1和X2，第三个是标签类别
    """
    data_mat = []
    label_mat = []
    # 读取格式：-0.017612	14.053064	0
    with open('testSet.txt', 'r') as f:
        for line in f.readlines():
            # 拆分放到arr里面
            line_arr = line.strip().split()
            # 第一位是1？
            data_mat.append([1., float(line_arr[0]), float(line_arr[1])])
            # 标签变成2
            label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(inx):
    # 那个s形的函数
    return 1.0 / (1 + np.exp(-inx))


def grad_ascent(data_mat, class_labels):
    """
    梯度上升
    :param data_mat:
    :param class_labels:
    :return:
    """
    # matrix格式
    data_matrix = np.mat(data_mat)  # 2D np array，列是特征，行是样本
    # 也是matrix格式，并站了起来
    label_mat = np.mat(class_labels).transpose()  # 转置，行变列
    m, n = np.shape(data_matrix)
    alpha = 0.001  # lr
    max_cycles = 500  # iter
    weights = np.ones((n, 1))
    # 保存weight
    weights_history = np.zeros((max_cycles, n))
    for k in range(max_cycles):
        # alpha = 0.004 / (1.0 + k) + 0.001
        h = sigmoid(data_matrix * weights)  # 是一个向量
        error = (label_mat - h)  # 真是简陋
        # 每次都撸一个进去
        weights = weights + alpha * data_matrix.transpose() * error
        weights_history[k, :] = weights.transpose()
    # 根本就不收敛
    return weights, weights_history


def stoc_grad_ascent(data_mat, class_labels, iter_counts=20):
    m, n = np.shape(data_mat)
    # alpha = 0.01  # 改进前
    weights = np.ones(n)
    # 迭代次数
    # iter_counts = 20
    # 存放历史
    weights_history = np.zeros((m * iter_counts, n))
    # 只跑那么多样本
    for j in range(iter_counts):
        data_index = list(range(m))
        for i in range(m):
            # 模拟退火常见？
            alpha = 4 / (1.0 + j + i) + 0.01  # 改进后，减少波动
            # 下面这种改动似乎不靠谱
            # rand_index = int(random.uniform(0,len(data_index)))
            h = sigmoid(np.sum(data_mat[i] * weights))
            error = class_labels[i] - h
            weights = weights + alpha * error * data_mat[i]
            weights_history[i + j * m, :] = weights
            # data_index.pop(rand_index)
    return weights, weights_history


def plot_data(weights):
    """
    画决策边界和散点图
    :param weights:
    :return:
    """
    import matplotlib.pylab as plt
    # 原本的weights是matrix，要转换为ndarray
    weights = np.squeeze(np.asarray(weights))
    data_mat, label_mat = load_dataset()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    xcord1, ycord1, xcord2, ycord2 = [], [], [], []
    for i in range(n):
        # 分两类
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    # 创建画板
    fig = plt.figure()
    # 设置布局
    ax = fig.add_subplot(111)
    # 散点图
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    # 连续的线段
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def plot_weights_update(weights_history):
    """用来画权重的更新图"""

    fig = plt.figure()
    # 三行一列的第一行
    ax = fig.add_subplot(311)
    type1 = ax.plot(weights_history[:, 0])
    plt.ylabel('X0')
    ax = fig.add_subplot(312)
    type2 = ax.plot(weights_history[:, 1])
    plt.ylabel('X1')
    ax = fig.add_subplot(313)
    type3 = ax.plot(weights_history[:, 2])
    plt.xlabel('iteration')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    data_arr, label_mat = load_dataset()  # 加载数据
    weights1, weights_history1 = grad_ascent(data_arr, label_mat)  # 梯度上升

    weights, weights_history = stoc_grad_ascent(np.array(data_arr), label_mat)  # 随机梯度

    plot_weights_update(weights_history)  # 随机
    plot_weights_update(weights_history1)  # 梯度
    plot_data(weights)  # 随机
    plot_data(weights1)  # 梯度
    pass
