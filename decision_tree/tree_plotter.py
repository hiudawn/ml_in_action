import matplotlib.pyplot as plt

# 定义文本框和箭头格式
# 文档找不到
# 第一个参数是形状，第二个不知道
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plot_node(node_text, center, parent, node_type):
    # create_plot.ax1 函数变量？？
    create_plot.ax1.annotate(node_text,xy=parent,xycoords='axes fraction',xytext=center,textcoords='axes fraction',
                             va='center',ha='center',bbox=node_type,arrowprops=arrow_args)

def create_plot(tree):
    # 创建新图形并清空绘图区
    figure = plt.figure(1,facecolor='white')
    figure.clf()
    # 下面三行就绘制的那个示例
    # create_plot.ax1 = plt.subplot(111,frameon = False)
    # plot_node('a decision node',(0.5,0.1),(0.1,0.5),decision_node)
    # plot_node('a leaf node',(0.8,0.1),(0.3,0.8),leaf_node)

    axprops = dict(xticks=[],yticks=[])
    create_plot.ax1 = plt.subplot(111,frameon = False,**axprops)
    plot_tree.total_w = float(get_num_leafs(tree))
    plot_tree.total_d = float(get_tree_depth(tree))
    plot_tree.x_off = -0.5/plot_tree.total_w
    plot_tree.y_off = 1.
    plot_tree(tree,(0.5,1.),'')

    plt.show()

def plot_mid_text(center,parent,txt_string):
    # 中间文本的坐标，上减下加上下
    x_mid = (parent[0]-center[0])/2. + center[0]
    y_mid = (parent[1]-center[1])/2. + center[1]
    create_plot.ax1.text(x_mid,y_mid,txt_string)

def plot_tree(tree,parent,node_text):
    """晦涩"""
    # 计算叶子数量
    num_leafs = get_num_leafs(tree)
    depth = get_tree_depth(tree)
    first_str = list(tree.keys())[0]
    # 定位
    center = (plot_tree.x_off + (1.0+float(num_leafs))/2./plot_tree.total_w,plot_tree.y_off)
    # 中间的文本
    plot_mid_text(center,parent,node_text)
    # 节点
    plot_node(first_str,center,parent,decision_node)
    second_dict = tree[first_str]
    plot_tree.y_off -= 1./plot_tree.total_d
    # 开始画了，也是递归
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key],center,str(key))
        else:
            plot_tree.x_off += 1./plot_tree.total_w
            plot_node(second_dict[key],(plot_tree.x_off,plot_tree.y_off),center,leaf_node)
            plot_mid_text((plot_tree.x_off,plot_tree.y_off),center,str(key))
    plot_tree.y_off += 1./plot_tree.total_d


def get_num_leafs(tree):
    """递归求叶子"""
    num_leafs = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        # 如果节点还是一个字典，就说明还可以继续
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            # 每次发现一个节点就加一，最终的那个子叶也是加个1就跑了
            num_leafs +=1

    return num_leafs

def get_tree_depth(tree):
    """其实差不太多"""
    max_depth = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

def retrieve_tree(i):
    # 先弄两个树来检索，省得一直创建
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0:'no', 1: 'yes'}},1:'no'}}}},
                     ]
    return list_of_trees[i]

def main():
    create_plot(retrieve_tree(0))
    print(get_num_leafs(retrieve_tree(0)))
    print(get_tree_depth(retrieve_tree(0)))

if __name__ == '__main__':
    main()