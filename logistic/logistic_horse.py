import numpy as np


def sigmoid(x):
    # 那个s形的函数
    return 1.0 / (1 + np.exp(-x))


def classify_vector(x, weights):
    # 概率超过一半就设为1
    prob = sigmoid(np.sum(x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


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


def colic_test():
    # 他说这个完全独立，有随机成分，这是因为它的随机梯度真的是随机的，我这个不随机
    train_set, train_labels = [], []
    with open('horseColicTraining.txt', 'r') as f:
        for line in f.readlines():
            # 每行的拆分
            train_list = line.strip().split()
            line_arr = []
            for i in range(21):
                # 把前面21个都放进去
                line_arr.append(float(train_list[i]))
            train_set.append(line_arr)
            # 最后一个是标签
            train_labels.append(float(train_list[21]))
    # 随机梯度下降学习
    train_weights, _ = stoc_grad_ascent(np.array(train_set), train_labels, 500)
    error_count = 0
    num_test_vec = 0.
    with open('horseColicTest.txt', 'r') as f:
        for line in f.readlines():
            # 看看有多少个测试数据
            num_test_vec += 1.
            test_list = line.strip().split()
            line_arr = []
            for i in range(21):
                line_arr.append(float(test_list[i]))
            # 测试数据结果是否正确
            if int(classify_vector(np.array(line_arr), train_weights)) != int(test_list[21]):
                error_count += 1
    # 总错误率
    error_rate = (float(error_count)) / num_test_vec
    print("the error rate of this test is: %f" % error_rate)
    return error_rate


if __name__ == '__main__':
    colic_test()
    pass
