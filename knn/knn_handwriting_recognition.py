import numpy as np
import os
import operator

def img2vector(filename):
    """想不通为什么要用文本形式存数字，真是zz"""
    ret_val = np.zeros((1,1024))
    with open(filename,'r') as f:
        lines = f.readlines()
    for i1,line in enumerate(lines):
        for i2,c in enumerate(line.strip()):
            ret_val[0,i1*32+i2] = int(c)
    return ret_val

def handwriting_class_test():
    labels = []
    # 返回该文件夹下文件或文件夹名字，升序
    trainling_file_list = os.listdir('./trainingDigits')
    # 初始化一个空的训练矩阵，放一个个数据
    training_mat = np.zeros((len(trainling_file_list),1024))
    # 把文件一个个提取
    for i,item in enumerate(trainling_file_list):
        # 某一个文件
        file_name_str = item
        # 文件去后缀
        file_str = file_name_str.split('.')[0]
        # 这个文件代表数字几
        class_num_str = int(file_str.split('_')[0])
        # 把数字标签放进去
        labels.append(class_num_str)
        # 把文件变成向量并填到矩阵的一行
        training_mat[i,:] = img2vector('./trainingDigits/%s' % file_name_str)
    # 大部分和上面类似
    test_file_list = os.listdir('./testDigits')
    error = 0.
    for i,item in enumerate(test_file_list):
        file_name_str = item
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('./testDigits/%s' % file_name_str)
        # 进行测试，k取3，在海量训练集中找最近的三个
        result = classify0(vector_under_test,training_mat,labels,3)
        # 想看匹配情况去掉注释
        # print("the classifier came back with: %d, the real answer is: %d"\
        #       % (result,class_num_str))
        # 分错了就计个数
        if(result != class_num_str) : error += 1.0
    # 显而易见
    print("the total number of errors is: %d" % error)
    print("the accuracy rate is: %f" % (1 - error/float(len(test_file_list))))

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

def save_to_img(arr):
    # 随便把某个向量化后的数字放进来就能保存成图片
    import scipy.misc
    scipy.misc.imsave('outfile.jpg', arr)

def main():
    # arr = img2vector('./testDigits/5_13.txt')
    # arr = np.reshape(arr,(32,32))
    # save_to_img(arr)
    handwriting_class_test()

if __name__ == '__main__':
    main()