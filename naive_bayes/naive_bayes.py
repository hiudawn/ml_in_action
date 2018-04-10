import random

import numpy as np
import re


def load_dataset():
    """
    创建了一些实验样本
    第一个变量返回切分集，第二个返回是否具有侮辱性
    """

    posting_list = [
        ['my','dog','has','flea',
        'problems','help','please'],
        ['maybe','not','take','him',
            'to','dog','park','stupid'],
        ['my','dalmation','is','so','cute',
         'I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how',
         'to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid']
    ]
    class_vec = [0,1,0,1,0,1]  # 1是侮辱，0正常
    return posting_list,class_vec

def create_vocab_list(dataset):
    # 创建一个空集,用来存所有的单词
    vocab_set = set([])
    for document in dataset:
        # 求两个集合的并集，或操作符
        vocab_set = vocab_set | set(document)
    return list(vocab_set)

def set_of_words2vec(vocab_list,input_set):
    """词集模型，词出现多少次都只记为1"""
    # 初始化一个长度为词汇表的0向量
    return_vec = [0]*len(vocab_list)
    # 对于输入集合中的某个词
    for word in input_set:
        # 如果这个词存在于词汇表中
        if word in vocab_list:
            # 计数器设为1
            return_vec[vocab_list.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return return_vec

def bag_of_words2vec(vocab_list,input_set):
    """朴素贝叶斯词袋模型"""
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            # 这里是加1
            return_vec[vocab_list.index(word)] += 1
    return return_vec

def train_nb0(train_matrix,train_category):
    """p1是侮辱性文档，，一旦某个词语侮辱或正常的，则对应的p1_num或p0_num就加1，再所有的文档中，该文档的总词语+1"""
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = np.sum(train_category)/float(num_train_docs)
    # p0_num = np.zeros(num_words)
    # p1_num = np.zeros(num_words)
    # # 分母的意思
    # p0_denom = 0.
    # p1_denom = 0.
    # 上面的初始化版本可能导致几个概率相乘为0，详见65页，改为：
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denom = 2.
    p1_denom = 2.
    for i in range(num_train_docs):
        # 如果文章是侮辱性的
        if train_category[i] == 1:
            # 向量的相加，即是侮辱性的，那么反正出现的词都叠加起来
            p1_num += train_matrix[i]
            p1_denom += np.sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += np.sum(train_matrix[i])
    # p1_vect = p1_num/p1_denom  # change to log(),因为很多很小的数相乘会导致下溢，而log没这个问题
    # p0_vect = p0_num/p0_denom  # change to log()
    p1_vect = np.log(p1_num / p1_denom)
    p0_vect = np.log(p0_num / p0_denom)
    # 返回这前两个向量的和都是1
    return p0_vect,p1_vect,p_abusive

def classify_nb(vec2classify,p0vec,p1vec,pclass1):
    # 输入：要分类的向量，用train_nb0计算得到的三个概率
    p1 = np.sum(vec2classify*p1vec) + np.log(pclass1)
    p0 = np.sum(vec2classify*p0vec) + np.log(1.0 - pclass1)
    # 返回分类结果
    if p1 > p0:
        return 1
    else:
        return 0

def testing_nb():
    # 获得数据
    list_of_posts,list_classes = load_dataset()
    # 提取出所有词库的唯一
    my_vocab_list = create_vocab_list(list_of_posts)
    # 用词向量来填充train_mat列表
    train_mat = []
    # 把那些句子都变成向量，塞进mat中
    for post_in_doc in list_of_posts:
        train_mat.append(set_of_words2vec(my_vocab_list,post_in_doc))
    # 对已经变成向量的句子，通过类标签来判别
    p0v,p1v,pab = train_nb0(np.array(train_mat),np.array(list_classes))
    # 进行测试
    test_entry = ['love','my','dalmation']
    this_doc = np.array(set_of_words2vec(my_vocab_list,test_entry))
    print(test_entry,'classified as: ',classify_nb(this_doc,p0v,p1v,pab))
    test_entry = ['stupid','garbage']
    this_doc = np.array(set_of_words2vec(my_vocab_list, test_entry))
    print(test_entry, 'classified as: ', classify_nb(this_doc, p0v, p1v, pab))

def split_text_test():
    my_sent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
    # 简单分割，有很多标点
    print(my_sent.split())
    # 我感觉是多打了一个\，就是用非字母来切分数据
    rtext = re.compile('\\W*')
    list_of_tokens = rtext.split(my_sent)
    print(list_of_tokens)
    t = [tok.lower() for tok in list_of_tokens if len(tok) > 0]
    email_text = open('email/ham/6.txt').read()
    list_of_tokens = rtext.split(email_text)
    print(list_of_tokens)

def text_parse(big_string):
    # 解析为字符串列表，去掉少于两个字符的字符串，并转为小写
    # 你想添加更多解析，就在这里搞
    list_of_tokens = re.split(r'\W*',big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]

def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    # 26会出错
    for i in range(1,23):
        # 读出所有的内容，调用parse打成词汇列表
        word_list = text_parse(open('email/spam/%d.txt' % i ).read())
        # 词汇列表放到doc里面
        doc_list.append(word_list)  # 列表中放了一个列表
        full_text.extend(word_list)  # 就是只有一个列表
        class_list.append(1)  # 这些是spam邮件
        # 同上
        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)  # 这些是ham邮件
    # 提取出每个词的一份
    vocab_list = create_vocab_list(doc_list)
    # training_set = list(range(50))写40会出错
    # 一个0到39的列表
    training_set = list(range(40))
    # 测试集
    test_set = []
    for i in range(10):
        # 随机构造一些测试集和训练集
        # 剩余部分还有交叉验证
        # 好弱，这么随便的随机
        rand_index = int(random.uniform(0,len(training_set)))
        # test取这些样本
        test_set.append((training_set[rand_index]))
        # 在训练集中干掉
        del(training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:
        # 变成1010那种，每一个doc慢慢练求10，求完了放进train中
        train_mat.append((set_of_words2vec(vocab_list,doc_list[doc_index])))
        # 记住对应的类
        train_classes.append((class_list[doc_index]))
    # 得到各种概率
    p0v,p1v,pspam = train_nb0(np.array(train_mat),np.array(train_classes))
    error_count = 0
    for doc_index in test_set:
        word_vec = set_of_words2vec(vocab_list,doc_list[doc_index])
        # 测试分类结果，错了+1
        # 上面那些概率就是来计算这个的
        if classify_nb(np.array(word_vec),p0v,p1v,pspam ) != class_list[doc_index]:
            error_count += 1
    print('the error rate is: ',float(error_count)/len(test_set))

if __name__ == '__main__':
    # list_classes是文章的侮辱与否标签
    # list_of_posts是简单的一些句拆分的词组
    # list_of_posts,list_classes = load_dataset()
    # # 提取出所有词库的唯一
    # my_vocab_list = create_vocab_list(list_of_posts)
    # print(my_vocab_list)
    # # 让一个句子拆分形成1 0 1 0 向量
    # a = set_of_words2vec(my_vocab_list,list_of_posts[0])
    # # 用词向量来填充train_mat列表
    # train_mat = []
    # # 把那些句子都变成向量，塞进mat中
    # for post_in_doc in list_of_posts:
    #     train_mat.append((set_of_words2vec(my_vocab_list,post_in_doc)))
    # # 对已经变成向量的句子，通过类标签来判别
    # p0v,p1v,pab = train_nb0(train_mat,list_classes)
    # print(p0v)
    # # 这里会有一个0.15789474左右的概率，正好对应stupid的下标，说明这个词是侮辱的概率很大
    # print(p1v)
    # print(pab)
    # ['love', 'my', 'dalmation'] classified as:  0
    # ['stupid', 'garbage'] classified as:  1
    # testing_nb()
    # split_text_test()
    spam_test()