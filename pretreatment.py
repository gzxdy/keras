

from math import log
import numpy as np
import pickle
import re
from gensim import corpora
import jieba
import xml.dom.minidom
from keras.preprocessing.sequence import pad_sequences




def read_xml(path):
    # 使用minidom解析器打开 XML 文档
    dom = xml.dom.minidom.parse(path)
    # dom = dom.documentElement
    root = dom.documentElement  # 看不了
    bs = root.getElementsByTagName("Doc")   #得到所有Doc节点
    st0 = ""
    listm = []
    listl = []
    lm = []
    lm1 = []
    for bd in bs:       #在每一个Doc节点中
        bb = bd.getElementsByTagName("Sentence")        #得到所有Sentence节点
        l = len(bb)

        for i in range(0, l):
            if bb[i].hasAttribute("label"):     #对于第i个Sentence节点，如果有label，则：
                # print ("+++++")
                if bb[i].getAttribute("label") == "0":
                    st0 = st0 + str(bb[i].childNodes[0].data) + " " + "\n"
                    lm.append(list(jieba.cut(str(bb[i].childNodes[0].data))))
                    lm1+=list(jieba.cut(str(bb[i].childNodes[0].data)))


                    listl.append([1,0,0])

                if bb[i].getAttribute("label") == "1":
                    st0 = st0 + str(bb[i].childNodes[0].data) + " " + "\n"
                    lm.append(list(jieba.cut(str(bb[i].childNodes[0].data))))
                    lm1 += list(jieba.cut(str(bb[i].childNodes[0].data)))

                    listl.append([0,1,0])

                if bb[i].getAttribute("label") == "2":
                    st0 = st0 + str(bb[i].childNodes[0].data) + " " + "\n"
                    lm.append(list(jieba.cut(str(bb[i].childNodes[0].data))))
                    lm1+=list(jieba.cut(str(bb[i].childNodes[0].data)))

                    listl.append([0,0,1])
    list0 = st0.split("\n")[:-1]
    return list0, np.mat(listl),lm,lm1      #一定要将列表转化成矩阵


def convert_doc_to_wordlist(str_doc, cut_all):
    #print(str_doc)
    # 分词的主要方法
    sent_list = str_doc.split('\n')  #列表，里面为一个字符串句子
    #print(sent_list)
    sent_list = map(rm_char, sent_list)  # 去掉一些字符，例如\u3000          map() 会根据提供的函数对指定序列做映射。
                            #第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。

    word_2dlist = [rm_tokens(jieba.cut(part, cut_all=cut_all)) for part in sent_list]  # 分词,得到表中表【【】】
    #print(word_2dlist)
    word_list = sum(word_2dlist, [])    #将word_2dlist中的元素与【】中的元素相加，相当于去掉word_2dlist外围的【】
    #print(word_list)
    return word_list

def rm_char(text):
    text = re.sub('\u3000', '', text)
    return text

def get_stop_words(path='./stopwork4.txt'):
    # stop_words中，每行放一个停用词，以\n分隔
    file = open(path, 'rb').read().decode('utf8').split('\n')
    return set(file)


def rm_tokens(words):  # 去掉一些停用次和数字
    words_list = list(words)
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words:  # 去除停用词
            words_list.pop(i)
        elif words_list[i].isdigit():   #判断是否为数字
            words_list.pop(i)
    return words_list



#list0, listl = read_xml()


#生成字典！！！！

def dict_te(list0):
    dictionary = corpora.Dictionary()
    #print(list0)
    for file in list0:
        file = convert_doc_to_wordlist(file, cut_all=False)     #file为列表，且已经经过分词、去停用词等操作，
                        # 如['-', '-', '-', '-', '天下足球', '《', '贝影', '(', '大卫', '·', '贝克汉姆', '传', ')', '》', ' ']
        #print(file)
        dictionary.add_documents([file])    #参数是表中表[[]]
    return dictionary
#dictionary.filter_n_most_frequent(10)
# dictionary.filter_extremes(no_below=3)
# p=list(dictionary.values())
# print("处理词典: ",p)
# # dictionary.compacity()
# o = input()

#dictionary = dict_te(list0)

def de_dictionary(dictionary):
    small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 10 ]
    dictionary.filter_tokens(small_freq_ids)
    dictionary.compactify()
    print(dictionary[len(dictionary)-1])
    return dictionary
    # print(dictionary[7207])


def dic_xy(l1,dictionary):
    a = []
    for i in l1:
        lxy = []
        for j in i:
            if j in dictionary.values():
                lxy.append(list(dictionary.keys())[list(dictionary.values()).index(j)])
        a.append(lxy)
    return a



def main():


    list0, listl,l1,c1 = read_xml("SMP2019_ECISA_Train.xml")   #解析xml,这里主要取list0，list1,list0是一个列表，里面是句子文本[txt1,txt2,txt3...]
    #list0为列表，里面为字符串句子
    #l1为列表，里面又是许多已经分词的句子的列表
    #print(l1)

    list00, listll, l11,c11 = read_xml("SMP2019_ECISA_Dev.xml")
    textss = l11


    dictionary = dict_te(list0)  # 将文本转化成词典

    print(dictionary)


    #print(sorted(list(dictionary.items()), key=lambda x: x[0]))

    dictionary = de_dictionary(dictionary)  #去低频词
    #print(list(dictionary.items()))


    dictionary =dict(sorted(list(dictionary.items()), key=lambda x: x[0])[1:]+[(len(dictionary),list(dictionary.values())[0])])

    #print(dict(sorted(list(dictionary.items()), key=lambda x: x[0])))
    print(dictionary)
    #i = input()
    a = dic_xy(l1,dictionary)
    c = dic_xy(l11, dictionary)


    lie = 100    #样本的列数

    X_train = pad_sequences(a, maxlen=lie, padding='pre')
    Y_train = listl
    # print("Y_train")
    # print(len(X_train),"---",len(Y_train))
    print(X_train)

    b=dic_xy(l11,dictionary)

    X_test = pad_sequences(b, maxlen=lie, padding='pre')
    Y_test = listll
    # print(len(X_test), "---", len(Y_test))
    with open("./data1","wb") as f:
        pickle.dump((X_train,Y_train,X_test,Y_test),f)

if __name__ == '__main__':
    main()

