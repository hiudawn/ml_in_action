import numpy as np
import util


def smo_simple(data_mat, class_labels, C, toler, max_iter):
    """

    :param data_mat:数据集
    :param class_labels:类别标签
    :param C:常数C
    :param toler: 容错率
    :param max_iter:最大循环次数
    :return:
    """
    data_matrix = np.mat(data_mat)
    label_mat = np.mat(class_labels).transpose()  # 列向量
    b = 0
    m, n = np.shape((data_matrix))
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < max_iter:
        alpha_pairs_changed = 0  # 每次都是0开始，用于记录是否优化
        for i in range(m):
            fxi = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            ei = fxi - float(label_mat[i])  # 误差
            if ((label_mat[i] * ei < -toler) and (alphas[i] < C) )or ((label_mat[i] * ei > toler) and (alphas[i] > 0)):
                j = util.select_j(i, m)
                fxj = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                ej = fxj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                # L = 100
                # H = 100
                if (label_mat[i] != label_mat[j]):
                    temp = float(alphas[j] - alphas[i])
                    L = max(0, temp)
                    H = min(C, temp + C)
                else:
                    temp = float(alphas[j] + alphas[i])
                    L = max(0, temp - C)
                    H = min(C, temp)
                # return
                if L == H:
                    # 不等不会退出循环？
                    print("L==H")
                    continue
                # 这是alpha[j]的最优修改量
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i,:].T - data_matrix[j, :] * data_matrix[j,:].T



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
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                b1 = b - ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
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
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


if __name__ == '__main__':
    data_arr, labels_arr = util.load_dataset('testSet.txt')
    b, alphas = smo_simple(data_arr, labels_arr, 0.6, 0.001, 40)
    print('b: ', b, 'alphas: ', alphas)
    print(alphas[alphas>0])
    # for i
    pass
