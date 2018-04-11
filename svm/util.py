import random

"""
这里求解的SVM优化问题用的是拉格朗日乘子法
参考机器学习实战95页或者李航统计学习104页
"""


def load_dataset(filename):
    data_mat, label_mat = [], []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line_arr = line.strip().split()
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[2]))
        return data_mat, label_mat


def select_j(i, m):
    """

    :param i:alpha下标
    :param m: 所有的alpha数目
    :return:
    """
    j = i
    while (j == i):
        # 随机生成下一个实数,左最小包含，右最大不包含
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    """

    :param aj:需要调整的alpha
    :param H: 上限
    :param L: 下限
    :return:
    """
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


if __name__ == '__main__':
    data_arr,label_arr = load_dataset('testSet.txt')
    print(label_arr)
    pass