import pandas as pd
import csv
import jieba
import jieba.analyse as analyse
from gensim import corpora
from gensim import models
from gensim.models import word2vec
import time
import matplotlib.pyplot as plt
import random




###############数据集预处理，把输入和输出分别保存到两个文件
#read csv,csv to list
def read_data(p):
    data=[]
    with open(p,'r') as f:
        reader=csv.reader(f)
        for row in reader:
            data.append(row)
    return data
#list to csv
def ltoc(d,p):
    pd.DataFrame(d).to_csv(p,quoting=1,header=False,encoding='utf-8',index=False)
############################################################################
#excel to csv
path='huaxiajiyin.xlsx'
data=pd.read_excel(path,index=False)
data.to_csv('huaxiajiyin.csv',quoting=1,header=False,encoding='utf-8',index=False)

data=read_data('huaxiajiyin.csv')
random.shuffle(data)

# #######################
def TIME(t):
    timeArray = time.strptime(t, " %Y-%m-%d %H:%M:%S ")
    if 0<=int(timeArray[3])<=3:
        return str('拂晓')
    if 3<int(timeArray[3])<=6:
        return str('黎明')
    if 6<int(timeArray[3])<=9:
        return str('清晨')
    if 9<int(timeArray[3])<=12:
        return str('上午')
    if 12<int(timeArray[3])<=15:
        return str('中午')
    if 15<int(timeArray[3])<=18:
        return str('下午')
    if 18<int(timeArray[3])<=21:
        return str('傍晚')
    if 21<int(timeArray[3])<=24:
        return str('深夜')
    if timeArray[3]=="":
        return str('未知')
def erfenlei(tar):
    for i in range(len(tar)):
        if 0<=tar[i]<30:
            tar[i] = 0
        elif 30<=tar[i]<=951:
            tar[i] = 1
    return tar
def read_txt(p):
    data = []
    with open(p,'r',encoding='gbk') as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            data.append(str(line))
    return data
stop_word = read_txt('stopwords.txt')
def word_seg(s):
    #jieba.add_word(stop_word[0])
    add=['汪建','姜广策','藻酸双酯钠']
    delate=['了','的']
    for x in add:
        jieba.add_word(x)
    for x in delate:
        jieba.del_word(x)
    str_list = list(jieba.cut(s,cut_all=False,HMM=False))
    s_list=[]
    for x in str_list:
        if x not in stop_word:
            s_list.append(x)
    return s_list
tar = []
a= []
Time=[]
source=[]
l=0
for i in data:
    i[3]=TIME(i[3])
    Time.append(i[3])
    source.append(i[6])
    tar.append(int(i[5])//1000+int(i[4]))

    i.pop(5)
    i.pop(4)
    i.pop(0)
    i.pop(0)
    s=''
    for k in i:
        s+=k
    i.append(s)
    i.pop(0)
    i.pop(0)
    i.pop(0)
    i.pop(0)

print(l,l//860)
a=sorted(tar)
print(a[430])
ltoc(data,'huaxiajiyin2.csv')
ltoc(erfenlei(tar),'target.csv')
ltoc(Time,'time.csv')
ltoc(source,'source.csv')
print(tar)

print(sum(tar),sum(tar[-86:]))

x=[x for x in range(1,861)]
y=tar[:860]
plt.plot(x,y,'ro')
plt.show()

############################################################################################################

#########word2vec做词向量

import pandas as pd
import csv
import jieba
import jieba.analyse as analyse
from gensim import corpora
from gensim import models
from gensim.models import word2vec
import time


def rem(m):
    bd = '～ ↓↑()1234567890《》%%.；>,.， `![]【\;''“,：./?？、 }{|!:-。”\u3000'
    n=[]
    for i in m:
        n.append(i)
        if i in bd:
            n.remove(i)
    return n

data=read_data('huaxiajiyin2.csv')
stop_word = read_txt('stopwords.txt')
source = read_data('source.csv')
for i in range(860):
    source[i]=source[i][0]

def word_seg(s):
    add=['汪建','姜广策','藻酸双酯钠','盘口']+source
    delate=['了','的']
    for x in add:
        jieba.add_word(x)
    for x in delate:
        jieba.del_word(x)
    str_list = list(jieba.cut(s,cut_all=False,HMM=False))
    s_list=[]
    for x in str_list:
        if x not in stop_word:
            s_list.append(x)
    return s_list
voc=[]
for i in data:
    voc+=rem(word_seg(i[0]))
model = models.Word2Vec([voc], size=200, sg=0, iter=5,min_count=5)
model.wv.save_word2vec_format("word2Vec" + ".bin", binary=True)

##################################################################################################################

##########构造模型

import os
import csv
import time
import datetime
import random
import json
import jieba

from collections import Counter
from math import sqrt

import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

def rem_train(m):
    bd = '～ ↓↑()1234567890《》%%.；>,.， `![]【\;''“,：./?？、 }{|!:-。”\u3000'
    n=''
    for i in m:
        if i not in bd:
            n += i
    return n
def rem_test(m):
    bd = '～ ↓↑()《》%%.；>,.， `![]【\;''“,：./?？、 }{|!:-。”\u3000'
    n=''
    for i in m:
        if i not in bd:
            n += i
    return n
def read_txt(p):
    data = []
    with open(p,'r',encoding='gbk') as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            data.append(line)
    return data
def read_data_train(p):
    data=[]
    with open(p,'r') as f:
        reader=csv.reader(f)
        for row in reader:
            n=rem_train(row[0])
            row=n[:]
            data.append(row)
    return data
def read_data_test(p):
    data=[]
    with open(p,'r') as f:
        reader=csv.reader(f)
        for row in reader:
            n = rem_test(row[0])
            row = n[:]
            data.append(row)
    return data
def word_seg(s):
    #tukai
    source = read_data_test('source.csv')
    for i in range(860):
        source[i] = source[i][0]
    add = ['汪建', '姜广策', '藻酸双酯钠']+source
    delate = ['了', '的']
    for x in add:
        jieba.add_word(x)
    for x in delate:
        jieba.del_word(x)
    str_list = list(jieba.cut(s,cut_all=False,HMM=False))
    return str_list

# 配置参数

class TrainingConfig(object):
    # 训练参数
    epoches = 40
    evaluateEvery = 100
    checkpointEvery = 100
    #learningRate = 0.001
    learningRate = 0.001

class ModelConfig(object):
    # 模型参数
    embeddingSize = 200
    numFilters = 128#每个尺寸的卷积核的个数都为128
    filterSizes = [2, 3, 4, 5]  # 卷积核size，卷积层只有一层
    dropoutKeepProb = 1.0
    l2RegLambda = 0.0  # L2正则化系数

class Config(object):
    #tukai
    sequenceLength = 400  # 取了所有序列长度的均值
    #sequenceLength = 400

    #batchSize = 128
    batchSize = 128

    dataSource = "huaxiajiyin.csv"

    stopWordSource = "stopwords.txt"  # 停用词表

    numClasses = 3

    rate = 0.9  # 训练集的比例

    training = TrainingConfig()

    model = ModelConfig()

# 实例化配置参数对象
config = Config()

# 数据预处理的类，生成训练集和测试集
class Dataset(object):
    def __init__(self, config):
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource

        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate

        self._stopWordDict = {}

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.wordEmbedding = None

        self._wordToIndex = {}
        self._indexToWord = {}

    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """

        # df = pd.read_csv(filePath)
        # labels = df["sentiment"].tolist()
        # review = df["review"].tolist()
        # reviews = [line.strip().split() for line in review]
        labels = read_data_test('target.csv')
        #tukai
        # l=[]
        # for i in labels:
        #     l.append(np.eye(3)[int(i)])
        # labels=l[:]
        # print(labels[0])

        review = read_data_train('huaxiajiyin2.csv')
        reviews = [line for line in review]
        #tukai
        for i in range(len(reviews)):
            reviews[i]=word_seg(reviews[i])
        #print(reviews[0])

        return reviews, labels

    def _reviewProcess(self, review, sequenceLength, wordToIndex):
        """
        将数据集中的每条评论用index表示
        wordToIndex中“pad”对应的index为0
        """

        reviewVec = np.zeros((sequenceLength))
        sequenceLen = sequenceLength

        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)


        for i in range(sequenceLen):
            if review[i] in wordToIndex:
                reviewVec[i] = wordToIndex[review[i]]
            else:
                reviewVec[i] = wordToIndex["UNK"]
        # tukai
        # print(review[0])
        # print(reviewVec[0])
        return reviewVec

    def _genTrainEvalData(self, x, y, rate):
        """
        生成训练集和验证集
        """

        reviews = []
        labels = []

        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x)):
            reviewVec = self._reviewProcess(x[i], self._sequenceLength, self._wordToIndex)
            reviews.append(reviewVec)

            labels.append([y[i]])

        trainIndex = int(len(x) * rate)

        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _genVocabulary(self, reviews):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """

        allWords = [word for review in reviews for word in review]

        # 去掉停用词
        subWords = [word for word in allWords if word not in self.stopWordDict]

        wordCount = Counter(subWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        #words = [item[0] for item in sortWordCount if item[1] >= 5]
        #tukai
        words = [item[0] for item in sortWordCount if item[1] >= 5]

        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding

        self._wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self._indexToWord = dict(zip(list(range(len(vocab))), vocab))

        #将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("wordToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self._wordToIndex, f)

        with open("indexToWord.json", "w", encoding="utf-8") as f:
            json.dump(self._indexToWord, f)

    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """
        #tukai
        wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2Vec.bin", binary=True)

        vocab = []
        wordEmbedding = []

        # 添加 "pad" 和 "UNK",
        vocab.append("pad")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))

        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")

        return vocab, np.array(wordEmbedding)

    def _readStopWord(self, stopWordPath):
        """
        读取停用词
        """

        stopWordList = read_txt(stopWordPath)
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
        self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    def dataGen(self):
        """
        初始化训练集和验证集
        """

        # 初始化停用词
        self._readStopWord(self._stopWordSource)

        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)

        # 初始化词汇-索引映射表和词向量矩阵
        self._genVocabulary(reviews)

        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviews, labels, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels


data = Dataset(config)
data.dataGen()

# 输出batch数据集
def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    #tukai
    #print(x,y)

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY


# 构建模型
class TextCNN(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, config, wordEmbedding):
        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        #tukai
        #self.inputY = tf.placeholder(tf.float32, [None, config.numClasses], name="inputY")
        self.inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)
            # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
            self.embeddedWordsExpanded = tf.expand_dims(self.embeddedWords, -1)

        # 创建卷积和池化层
        pooledOutputs = []
        # 有三种size的filter，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
        for i, filterSize in enumerate(config.model.filterSizes):
            with tf.name_scope("conv-maxpool-%s" % filterSize):
                # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
                # 初始化权重矩阵和偏置
                filterShape = [filterSize, config.model.embeddingSize, 1, config.model.numFilters]
                W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[config.model.numFilters]), name="b")
                conv = tf.nn.conv2d(
                    self.embeddedWordsExpanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # relu函数的非线性映射
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, config.sequenceLength - filterSize + 1, 1, 1],
                    # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooledOutputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中

        # 得到CNN网络的输出长度
        numFiltersTotal = config.model.numFilters * len(config.model.filterSizes)

        # 池化后的维度不变，按照最后的维度channel来concat
        self.hPool = tf.concat(pooledOutputs, 3)

        # 摊平成二维的数据输入到全连接层
        self.hPoolFlat = tf.reshape(self.hPool, [-1, numFiltersTotal])

        # dropout
        with tf.name_scope("dropout"):
            self.hDrop = tf.nn.dropout(self.hPoolFlat, self.dropoutKeepProb)

        # 全连接层的输出
        with tf.name_scope("output"):
            #tukai
            outputW = tf.get_variable(
                "outputW",
                shape=[numFiltersTotal, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            outputB = tf.Variable(tf.constant(0.1, shape=[1]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.predictions = tf.nn.xw_plus_b(self.hDrop, outputW, outputB, name="predictions")
            self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.0), tf.float32, name="binaryPreds")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            #tukai
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
            print(self.predictions,' ',self.inputY)


            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss


# 定义性能指标函数

def mean(item):
    return sum(item) / len(item)


def genMetrics(trueY, predY, binaryPredY):
    """
    生成acc和auc值
    """
    auc = roc_auc_score(trueY, predY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY)
    recall = recall_score(trueY, binaryPredY)


    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)


# 训练模型

# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

wordEmbedding = data.wordEmbedding

# 定义计算图
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

    sess = tf.Session(config=session_conf)

    # 定义会话
    with sess.as_default():
        cnn = TextCNN(config, wordEmbedding)

        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(cnn.loss)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        print("Writing to {}\n".format(outDir))

        lossSummary = tf.summary.scalar("loss", cnn.loss)
        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)


        sess.run(tf.global_variables_initializer())


        def trainStep(batchX, batchY):
            """
            训练函数
            """
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                #tukai
                cnn.dropoutKeepProb: config.model.dropoutKeepProb
            }
            _, summary, step, loss, predictions, binaryPreds = sess.run(
                [trainOp, summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                feed_dict)
            timeStr = datetime.datetime.now().isoformat()
            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)
            if (i+1)%m==0:
                print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(timeStr, step, loss, acc,
                                                                                               auc, precision, recall))
            trainSummaryWriter.add_summary(summary, step)


        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: 1.0
            }

            summary, step, loss, predictions, binaryPreds = sess.run(
                [summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                feed_dict)

            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)

            evalSummaryWriter.add_summary(summary, step)
            #tukai
            return loss, acc, auc, precision, recall,binaryPreds,predictions


        for i in range(config.training.epoches):
            # 训练模型
            n=1
            m=10
            if (i+1)%n==0:
                if (i + 1) % m == 0:
                    print("start training model", '->', i)
                for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                    trainStep(batchTrain[0], batchTrain[1])

                    currentStep = tf.train.global_step(sess, globalStep)
                    if currentStep % ((len(trainLabels) // config.batchSize)*m) == 0:
                        print("\nEvaluation:")

                        losses = []
                        accs = []
                        aucs = []
                        precisions = []
                        recalls = []
                        #tukai
                        result=[]
                        re=[]
                        for batchEval in nextBatch(evalReviews, evalLabels, len(evalLabels)):
                            loss, acc, auc, precision, recall,binaryPreds,predictions = devStep(batchEval[0], batchEval[1])
                            losses.append(loss)
                            accs.append(acc)
                            aucs.append(auc)
                            precisions.append(precision)
                            recalls.append(recall)
                            #tukai
                            result.append(binaryPreds)
                            re.append(predictions)
                        time_str = datetime.datetime.now().isoformat()
                        print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(time_str,
                                                                                                           currentStep,
                                                                                                           mean(losses),
                                                                                                           mean(accs),
                                                                                                           mean(aucs),
                                                                                                           mean(precisions),
                                                                                                         mean(recalls)))
