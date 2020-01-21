# -*-coding:utf-8-*-

import re
import tensorflow as tf
# from gensim.models import word2vec
from string import punctuation
import pandas as pd
import jieba
import codecs
import numpy as np
import datetime
import random
# import uniout


def load_sentences(path):

    with codecs.open(path, encoding='utf-8-sig') as f:
        text = f.read()
    # print text
    lines = text.split('\n')

    return lines


def get_train_batch():

    labels = []
    arr = np.zeros([batch_size, MaxSeqNum])
    for i in range(batch_size):
        if i % 2 == 0:
            num = random.randint(0, len(PositiveSentence) - 60)
            labels.append([1, 0])
        else:
            num = random.randint(len(PositiveSentence) + 60, len(AllSentence) - 1)
            labels.append([0, 1])
        arr[i] = ids[num]

    return arr, labels


# 同上
def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, MaxSeqNum])
    for i in range(batch_size):
        num = random.randint(len(PositiveSentence) - 60, len(PositiveSentence) + 60)
        if num <= len(PositiveSentence):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num]
    return arr, labels


if __name__ == "__main__":

    MaxSeqNum = 60
    num_dimensions = 100
    Path1 = 'C:/Users/text/Desktop/data_news/'
    # model = word2vec.Word2Vec.load(Path1 + 'sentences3.model')

    WordVector = pd.read_csv(Path1 + "wordvector2.csv").set_index('Unnamed: 0')
    WordVector = WordVector.values
    print (WordVector.shape)
    WordVector = WordVector.astype(np.float32)

    PositiveSentence = load_sentences(Path1 + "positive_cut2.txt")
    NegativeSentence = load_sentences(Path1 + "negative_cut2.txt")
    AllSentence = PositiveSentence + NegativeSentence
    print (len(PositiveSentence), len(NegativeSentence), len(AllSentence))
    WordList = ["".join(re.findall('\S', CleanWord)) for CleanWord in load_sentences(Path1 + "wordid2.txt")]
    print (len(WordList))
    print (WordList[:10])

    ids = np.zeros((len(AllSentence), MaxSeqNum), dtype='int32')
    for NumOfSentence, sentence in enumerate(AllSentence):
        for NumOfWord, word in enumerate(sentence.replace(u"\n", "").split(" ")):
            if NumOfWord >= MaxSeqNum:
                break

            try:
                ids[NumOfSentence][NumOfWord] = WordList.index(word)

            except ValueError:
                ids[NumOfSentence][NumOfWord] = 21205

        # print (NumOfSentence, ids[NumOfSentence])

    np.save(Path1 + "ids_matrix2.npy", ids)

    batch_size = 100  # batch的尺寸
    lstm_units = 64  # lstm的单元数量
    num_labels = 2  # 输出的类别数
    iterations = 20005  # 迭代的次数
    # 载入正负样本的词典映射
    ids = np.load(Path1 + 'ids_matrix2.npy')
    # for id in ids:
        # print (np.array(WordList)[id])
    print(u'载入IDS:', ids.shape)
    # for i in ids:
        # print (i)

    tf.reset_default_graph()
    # 确定好单元的占位符：输入是24x120，输出是24x2
    labels = tf.placeholder(tf.float32, [batch_size, num_labels])
    input_data = tf.placeholder(tf.int32, [batch_size, MaxSeqNum])

    # 必须先定义该变量
    # data = tf.Variable(tf.zeros([batch_size, MaxSeqNum, num_dimensions]), dtype=tf.float32)
    # 调用tf.nn.lookup()接口获得文本向量，该函数返回batch_size个文本的3D张量，用于后续的训练
    data = tf.nn.embedding_lookup(WordVector, input_data)

    # 使用tf.contrib.rnn.BasicLSTMCell细胞单元配置lstm的数量
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    # 配置dropout参数，以此避免过拟合
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
    # 最后将LSTM cell和数据输入到tf.nn.dynamic_rnn函数，功能是展开整个网络，并且构建一整个RNN模型
    # 这里的value认为是最后的隐藏状态，该向量将重新确定维度，然后乘以一个权重加上偏置，最终获得label
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_labels]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    # 定义正确的预测函数和正确率评估参数
    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    # 最后将标准的交叉熵损失函数定义为损失值，这里是以adam为优化函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # tf.summary.scalar('loss', loss)
    # tf.summary.scalar('Accrar', accuracy)
    # merged = tf.summary.merge_all()
    # logdir = 'tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    # writer = tf.summary.FileWriter(logdir, sess.graph)

    for i in range(iterations):
        # 下个批次的数据
        next_batch, next_batch_labels = get_train_batch()
        sess.run(optimizer, {input_data: next_batch, labels: next_batch_labels})

        # 每50次写入一次leadboard
        # if i % 50 == 0:
            # summary = sess.run(merged, {input_data: next_batch, labels: next_batch_labels})
            # writer.add_summary(summary, i)

        if i % 1000 == 0:
            loss_ = sess.run(loss, {input_data: next_batch, labels: next_batch_labels})

            accuracy_ = (sess.run(accuracy, {input_data: next_batch, labels: next_batch_labels})) * 100
            print("iteration:{}/{}".format(i + 1, iterations),
                  "\ntrain loss:{}".format(loss_),
                  "\ntrain accuracy:{}".format(accuracy_))

            test_batch, test_batch_labels = get_test_batch()
            loss_ = sess.run(loss, {input_data: test_batch, labels: test_batch_labels})

            accuracy_ = (sess.run(accuracy, {input_data: test_batch, labels: test_batch_labels})) * 100
            print("\ntest loss:{}".format(loss_),
                  "\ntest accuracy:{}".format(accuracy_))
            print (datetime.datetime.now().strftime("%H:%M:%S"))
            print('..........')

        # 每5000次保存一下模型
        if i % 5000 == 0 and i != 0:
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt")
            print("saved to %s" % save_path)

    # writer.close()
