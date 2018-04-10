import math
import operator
import tree_plotter

def calc_shannon_entropy(dataset):
    """
    计算香农熵，似乎是全类的标记
    熵越高，混合的数据越多
    另一种度量如基尼不纯度，简单来讲就是随机选子项度量误分概率
    """
    label_counts = {}
    for feature_vec in dataset:
        # 提取的就是那个分类的结果
        current_label = feature_vec[-1]
        # 计算一下各种结果一共有多少个
        label_counts[current_label] =  label_counts.get(current_label,0) + 1
    # 香农熵
    shannon_entropy = 0.
    for key in label_counts.keys():
        # 计算每一种结果的概率
        prob = float(label_counts[key]) / len(dataset)
        # 计算它的香农熵
        shannon_entropy -= prob * math.log(prob,2)
    return shannon_entropy

def create_dataset():
    """
    简单版的创建数据
    数据是列表，各个特征长度一致，最后一个是类别标签
    """
    # 可以把其中一个换成'maybe'试一试，熵明显会变大
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no'],
               ]
    labels = ['no surfacing', 'flippers']
    return  dataset, labels

def split_dataset(dataset,axis,value):
    """
    划分数据集
    """
    ret_dataset = []
    for feature_vec in dataset:
        if feature_vec[axis] == value:
            # 反正这这两句的意思就是剔除[axis]对应的元素
            reduced_feature_vec = feature_vec[:axis]
            reduced_feature_vec.extend(feature_vec[axis+1:])
            # 把剔除后的特征向量放进集合中
            ret_dataset.append(reduced_feature_vec)
    return ret_dataset

def choose_best_feature_to_split(dataset):
    """
    熵的计算理解：按各自占的比例一直往下算，最后一步shannon_entropy -= prob * math.log(prob,2)
    都计算完了再根据权值加回来
    返回划分数据集最佳的特征
    """
    # 减去类别标签，看有几个特征
    num_features = len(dataset[0]) - 1
    # 无序时候的基本熵
    base_entropy = calc_shannon_entropy(dataset)
    # 先任意初始化最佳信息增益和与之对应的最佳特征
    best_info_gain = 0.
    best_feature = -1
    # 尝试用每一个特征来划分
    for i in range(num_features):
        feature_list = [example[i] for example in dataset]
        # 每种分类去掉重复的特征
        unique_vals = set(feature_list)
        new_entropy = 0.
        # 对每一个唯一的特征划分数据集，计算新熵，对所有唯一的熵求和
        for value in unique_vals:
            # 开始划分
            sub_dataset = split_dataset(dataset, i, value)
            # 计算概率和新的熵
            prob = len(sub_dataset)/float(len(dataset))
            new_entropy += prob*calc_shannon_entropy(sub_dataset)
        # 信息增益是原本的熵-新的？
        info_gain = base_entropy - new_entropy
        # 假如熵大大减少了
        if(info_gain > best_info_gain):
            # 设定这个信息增益最佳，锁定最佳特征
            best_info_gain = info_gain
            best_feature = i
    # 返回的是最佳特征
    return best_feature

def majority_cnt(class_list):
    """多数表决，用于节点没有特征却有多分类的情况"""
    class_count = {}
    # 计算一下频率
    for vote in class_list:
        class_count[vote] = class_count.get(vote,0) + 1
    # 根据频率排序
    sorted_class_count = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    # 取频率最高的那个的第一个特征
    return sorted_class_count[0][0]

def create_tree(dataset,labels):
    """主方法"""
    # 取出类标签
    class_list = [example[-1] for example in dataset]
    # 如果数据中只含有一种类标签，返回该类
    if class_list.count(class_list[0]) == len (class_list):
        return class_list[0]
    # 如果数据没有特征了，但类别还有多种（因为上一关筛选掉了单种的），就进行多数表决
    if len (dataset[0]) == 1:
        return majority_cnt(class_list)
    # 选择最好的划分特征，并几下这个分类
    best_feature = choose_best_feature_to_split(dataset)
    # 这个特征叫什么名字
    best_feature_label = labels[best_feature]
    # 如果用的是递归，最内层就是最底层的那个分类
    tree = {best_feature_label:{}}
    # 用过这个特征了就删掉，不然会有重复
    del(labels[best_feature])
    # 相同特征只取一份
    feature_values = [example[best_feature] for example in dataset]
    unique_vals = set(feature_values)
    # 开始往底层递归了
    for value in unique_vals:
        # 除去刚那个已经删掉的特征剩下的子集
        sub_labels = labels[:]
        # 对子树进行操作
        tree[best_feature_label][value] = create_tree(split_dataset(dataset,best_feature,value),sub_labels)
    return tree

def store_tree(tree,filename):
    # 序列化，写到本地磁盘文件
    import pickle
    with open(filename,'wb') as f:
        pickle.dump(tree,f)

def grab_tree(filename):
    # 反序列化，从本地文件读出原有的对象
    import pickle
    with open(filename,'rb') as f:
        return pickle.load(f)

def classify(tree,feature_labels,test_vec):
    """
    第一个节点为当前母节点，往下找子节点，找到最后就返回
    """
    first_str = list(tree.keys())[0]
    # 字典的值，下面主要是判断这个还是不是字典，不是就是最终类标签
    second_dict = tree[first_str]
    # 找特征标签中对应这个键的序号
    feature_index = feature_labels.index(first_str)
    # 如果第二个字典的值还是字典
    for key in second_dict.keys():
        # 判断是0还是1
        if test_vec[feature_index] == key:
            print(key)
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key],feature_labels,test_vec)
            else:
                class_label = second_dict[key]
    return class_label

def load_glasses():
    with open('./lenses.txt','r') as f:
        lenses = [inst.strip().split('\t') for inst in f.readlines()]
        lenses_labels = ['age','prescript','astigmatic','tear_rate']
        return lenses,lenses_labels

def main():
    # dataset,labels = create_dataset()
    # print(dataset[0])
    # print(calc_shannon_entropy(dat))
    # a = split_dataset(dataset, 0, 1)
    # b = split_dataset(dataset, 0, 0)
    # c = choose_best_feature_to_split(dataset)
    # t = create_tree(dataset,labels)
    # t = tree_plotter.retrieve_tree(0)
    #
    # print(classify(t,labels,[0,0]))
    # print(classify(t,labels,[0,1]))
    # tree_plotter.create_plot(t)
    # print(t)
    # store_tree(t,'tree.txt')
    # a = grab_tree('tree.txt')
    # print(a)
    lenses,lenses_labels = load_glasses()
    t2 = create_tree(lenses,lenses_labels)
    print(t2)
    tree_plotter.create_plot(t2)

if __name__ == '__main__':
    main()