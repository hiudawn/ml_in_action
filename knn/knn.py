import numpy as np
# 运算符模块，这里主要用来排序
import operator
import matplotlib.pylab as plt

def create_dataset():
    group = np.array([[1.,1.1],[1.,1.],[0.,0.],[0.,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(in_x,dataset,labels,k):
    """简单分类器1"""
    # 行，也即点的个数，每个点一行表示，提取几行就几个点
    dataset_size = dataset.shape[0]
    # 纵向拓展一下输入的x，让它可以和原来的所有向量批量求差
    diff_mat = np.tile(in_x,(dataset_size,1)) - dataset

    # 求差的平方和
    sq_diff_mat = diff_mat**2
    # 在计算距离的时候，如果有个维度的数值很大，那么将直接影响计算的结果
    # 简单归一化new_value = (old_value - min)/(max-min)
    distance = (sq_diff_mat.sum(axis=1))**0.5 # 求与每一个点的距离平方，是列向量

    # 返回数组从小到大的索引，ndarray
    sorted_distance_indices = distance.argsort()
    # 弄个空的
    class_count = {}

    for i in range(k):
        # 把刚距离由小到大的label放进去
        vote_label = labels[sorted_distance_indices[i]]
        # 对每个label进行计数
        # dict.get前面是key，不存在赋0，存在返回原有的值
        class_count[vote_label] = class_count.get(vote_label,0)+1
    # 定义一个排序，排的是计数的所有内容，第二个域（第二维变量）
    # 改成operator.itemgetter(1，0)就会参造第二维先，再考虑第一维
    sorted_class_count = sorted(class_count.items(),
    key = operator.itemgetter(1),reverse=True)
    # 这里得到的B是[('B', 2), ('A', 1)]

    # 这个对应的就是标签号了
    return sorted_class_count[0][0]

def file2matrix(filename):
    """
    传入文件，读取内容，得到点和label
    第二列是玩游戏所耗时间比
    第三列是每周所消费的冰淇淋公升数
    """
    with open(filename,'r') as f:
        # 所有行数读取存在一个列表里
        lines = f.readlines()
    num_of_lines = len(lines)
    # 初始化点的矩阵,第二个参数其实直接可以写3，不过万一你要加多几列呢
    mat = np.zeros((num_of_lines,lines[0].count('\t')))
    # 初始化标签
    class_label_vector = []
    # 用enumerate省得再打一个index
    for index,line in enumerate(lines):
        # 每一行拆成列表，strip()去掉头尾任意空字符
        list_from_line = line.strip().split('\t')
        # 点
        mat[index, :] = list_from_line[0:3]
        # 标签,注这里的标签可能是string，看输入的是datingTestSet几
        class_label_vector.append(int(list_from_line[-1]))
    return mat,class_label_vector

def plot_figure(data_mat, labels):
    """
    没有加入legend
    有需要参考：https://www.zhihu.com/question/37146648
    """
    # 生成一个新的图像
    fig = plt.figure()
    # 这里的111的意思就是，图像画成一行一列（其实就一个框），最后一个1就是放在从左到右，从上到下的第1个
    # 想在一个画面里面放多几个子图和分配位置改下这个参数就好了
    ax = fig.add_subplot(111)
    # 前两个参数试一试data_mat[:,1],data_mat[:,2]
    # 或者data_mat[:,1],data_mat[:,0]  # 这种的区分度更高
    # scatter是画散点图
    # 10是点的大小，前后是颜色
    t = ax.scatter(data_mat[:,1],data_mat[:,0],
               10.0 * np.array(labels),np.array(labels))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.title(u'散点图')
    plt.xlabel(u'打机时间')
    plt.ylabel(u'飞机里程')
    plt.show()

def plot_figure2(data_mat, labels):
    """
    把每个类拆开的版本
    """
    print(data_mat[np.array(labels) == 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    type1_x = data_mat[np.array(labels) == 1][:, 1]
    type1_y = data_mat[np.array(labels) == 1][:, 0]
    type2_x = data_mat[np.array(labels) == 2][:, 1]
    type2_y = data_mat[np.array(labels) == 2][:, 0]
    type3_x = data_mat[np.array(labels) == 3][:, 1]
    type3_y = data_mat[np.array(labels) == 3][:, 0]

    t1 = ax.scatter(type1_x,type1_y,1,'r')
    t2 = ax.scatter(type2_x,type2_y,1,'g')
    t3 = ax.scatter(type3_x,type3_y,1,'b')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.title(u'散点图')
    plt.xlabel(u'打机时间')
    plt.ylabel(u'飞机里程')
    ax.legend((t1, t2, t3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2)
    plt.show()

def auto_norm(dataset):
    # 类似[0.       0.       0.001156]
    min_val = np.min(dataset,axis=0)
    # [9.1273000e+04 2.0919349e+01 1.6955170e+00]
    max_val = np.max(dataset,axis=0)
    # 范围
    ranges = max_val - min_val
    # 行数，即样本数
    m = dataset.shape[0]
    # 归一化
    norm_dataset = (dataset - np.tile(min_val,(m,1)))/ranges
    return norm_dataset,ranges,min_val

def dating_class_test():
    ratio = 0.1
    mat,labels = file2matrix('./datingTestSet2.txt')
    # 归一化
    norm_mat,ranges,miv_val = auto_norm(mat)
    # 样本数
    m = norm_mat.shape[0]
    # 取一定的样本
    num_test_vecs = int(m*ratio)
    error = 0.
    for i in range(num_test_vecs):
        # 只能一个个取，真弱，knn算了，不然有空改下claasify
        result = classify0(norm_mat[i,:],norm_mat[num_test_vecs:m,:],
                           labels[num_test_vecs:m],3)
        # print("the classifier came back with: %d, the real answer is: %d"\
        #       % (result,labels[i]))
        if result != labels[i]:
            error+=1.
    print("the accuracy rate is: %.1f%%" % (100*(1 - error/float(num_test_vecs))))

def main():
    group,labels = create_dataset()
    # 看看点[0,0]的归类，k取值3
    result = classify0([0,0],group,labels,3)
    print('the class of [0,0] is: ',result)
    mat,labels = file2matrix('./datingTestSet2.txt')
    plot_figure2(mat, labels)
    norm_dataset, ranges, min_val = auto_norm(mat)
    # 画散点图
    plot_figure2(norm_dataset, labels)
    # 预测
    dating_class_test()

if __name__ == '__main__':
    main()
