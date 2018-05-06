import numpy as np
import util
import matplotlib.pyplot as plt

"""
最初版为线性可分SVM，用的是SMO简略版本
"""


def smo_simple(data_mat, class_labels, C, toler, max_iter):
    """
    从早上八点改造到下午三点，叹气
    :param data_mat:数据集
    :param class_labels:类别标签
    :param C:常数C
    :param toler: 容错率
    :param max_iter:最大循环次数
    :return:
    """
    # [[3.542485e+00  1.977398e+00]
    #  [3.018896e+00  2.556416e+00]...
    data_matrix = np.array(data_mat)
    label_mat = np.array(class_labels).T
    label_mat = label_mat.reshape(-1, 1)  # 把简单的列表转化为列向量
    b = 0
    m, n = np.shape((data_matrix))  # 样本个数,维度
    # [[0.]
    #  [0.]...
    alphas = np.array(np.zeros((m, 1)))  # 样本个数，每个样本点对应有一个alpha
    iter = 0
    while iter < max_iter:
        alpha_pairs_changed = 0  # 用于记录是否优化
        # 对样本个数进行迭代
        for i in range(m):
            # 预测的类别
            fxi = float(np.dot((alphas * label_mat).T, (np.dot(data_matrix, data_matrix[i, :].T.reshape(-1, 1))))) + b
            # 和真实值的误差
            ei = fxi - float(label_mat[i])  # 误差
            # 如果预测得不太好
            if ((label_mat[i] * ei < -toler) and (alphas[i] < C)) or ((label_mat[i] * ei > toler) and (alphas[i] > 0)):
                # 随机选个alpha j值
                j = util.select_j(i, m)
                # 也预测一下
                fxj = float(
                    np.dot((alphas * label_mat).T, (np.dot(data_matrix, data_matrix[j, :].T.reshape(-1, 1))))) + b
                # 得到误差
                ej = fxj - float(label_mat[j])
                # 记下原本的alpha值
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                # 假设这两个预测出来的不一样，调整到0和C之间
                if (label_mat[i] != label_mat[j]):
                    # 设置一个临时变量
                    temp = float(alphas[j] - alphas[i])
                    # 下界
                    L = max(0, temp)
                    # 上界
                    H = min(C, C + temp)
                else:
                    # 如果相等就是俩alpha加起来
                    temp = float(alphas[j] + alphas[i])
                    L = max(0, temp - C)
                    H = min(C, temp)
                # 上下界一样了，退出更新
                if L == H:
                    # 不等不会退出循环？
                    print("L==H")
                    continue
                # 这是alpha[j]的最优修改量
                eta = 2.0 * np.dot(data_matrix[i, :], data_matrix[j, :].T) - np.dot(data_matrix[i, :],
                                                                                    data_matrix[i, :].T) - np.dot(
                    data_matrix[j, :], data_matrix[j, :].T)

                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= label_mat[j] * (ei - ej) / eta
                # 辅助函数随机选第二个alpha
                alphas[j] = util.clip_alpha(alphas[j], H, L)
                # 新老alpha比较
                if (np.abs(alphas[j] - alpha_j_old) < 0.00001):
                    print("j not moving enough")
                    continue
                # 修改方向和alpha j相反，大小一样
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                # 分别为他们两个设置各自的常数项
                b1 = b - ei - np.dot(np.dot(np.dot(label_mat[i], (alphas[i] - alpha_i_old)), data_matrix[i, :]),
                                     data_matrix[i, :].T) - \
                     np.dot(np.dot(np.dot(label_mat[j], (alphas[j] - alpha_j_old)), data_matrix[i, :]),
                            data_matrix[j, :].T)
                b2 = b - ei - np.dot(np.dot(np.dot(label_mat[i], (alphas[i] - alpha_i_old)), data_matrix[i, :]),
                                     data_matrix[j, :].T) - \
                     np.dot(np.dot(np.dot(label_mat[j], (alphas[j] - alpha_j_old)), data_matrix[j, :]),
                            data_matrix[j, :].T)
                # 把alpha调到0和C之间
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j] and C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print("iter: %d i: %d, pairs changed %d" % (iter, i, alpha_pairs_changed))
        if (alpha_pairs_changed == 0):
            # 木有任何改动的话
            iter += 1
        else:
            # 如果发生改变，就一直是0
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


def plot(data_arr, labels_arr):
    # 先转化为np对象，方便画图
    data_arr = np.array(data_arr)
    labels_arr = np.array(labels_arr)
    # 通过smo求得拉格朗日乘子法中的alpha和超平面的截距的b
    # 详细的公式可见李航统计学习方法P104页中的式7.19和7.26
    b, alphas = smo_simple(data_arr, labels_arr, 0.6, 0.001, 40)

    # 正样本
    data_arr_1 = data_arr[labels_arr == 1]
    # 负样本
    data_arr_n1 = data_arr[labels_arr == -1]
    # 支持向量，在0和C之间
    spp_vec = []
    for i in range(100):
        if alphas[i] > 0 and alphas[i] < 0.6:
            spp_vec.append(data_arr[i])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 准备画边界线
    x = np.arange(2, 7, 0.3)
    w = np.sum(alphas * labels_arr.reshape(-1, 1) * data_arr, axis=0)
    # 因为是二维的，所以画图和y = ax+b差不多，斜率就是-w0/w1
    plt.plot(x, (-w[0] * x - b) / w[1])
    ax.scatter(data_arr_1[:, 0], data_arr_1[:, 1], s=30, c='red', marker='s')
    ax.scatter(data_arr_n1[:, 0], data_arr_n1[:, 1], s=30, c='green')
    # print(spp_vec)
    # 画出空心的
    ax.scatter([i[0] for i in spp_vec], [i[1] for i in spp_vec], s=100, c='', marker='o', edgecolors='g')

    plt.show()


def main():
    data_arr, labels_arr = util.load_dataset('testSet.txt')
    b, alphas = smo_simple(data_arr, labels_arr, 0.6, 0.001, 40)
    print('b: ', b, 'alphas: ', alphas) 
    # 0元素太多了，就打印几个就好了，SMO算法是随机的
    # 这种过滤只对numpy有用
    print(alphas[alphas > 0])
    # 画出支撑向量
    for i in range(100):
        if alphas[i] > 0:
            print('support vector: ',data_arr[i], labels_arr[i])


if __name__ == '__main__':
    # main()
    data_arr, labels_arr = util.load_dataset('testSet.txt')
    plot(data_arr, labels_arr)
    pass
