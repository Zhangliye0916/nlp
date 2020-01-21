# -*-coding:utf-8-*-
# python 2.7
import re
import jieba
import uniout


def stop_words(path):

    return [line.strip().decode('utf-8') for line in open(path, 'r').readlines()]  # 读入停用词


def seg_words(sentence, stopwords):

    return list(filter(lambda x: x not in stopwords, jieba.cut(sentence)))


def pouncts():

    return [u'.', u'。', u',', u'，', u'!', u'！', u'?', u'？', u';', u'；']


def seg_sentences(sentence, pounct):

    return re.split('[' + '|'.join(pounct) + ']', sentence)


def load_words(path):

    with open(path) as f:
        text = f.readlines()
        my_dict = {}

        for line in text:
            word, weight = line.replace('\n', '').split('	')
            # print word, weight
            try:
                my_dict[word.decode('utf-8')] = float(weight)

            except:
                continue

        return my_dict


def laod_sent(path):

    with open(path) as f:
        text = f.readlines()
        my_dict = {}

        for line in text:
            index, sentence = line.replace('\n', '').split('	')
            my_dict[index] = sentence

        return my_dict


if __name__ == "__main__":

    Path1 = 'C:/Users/text/Desktop/data_news/'
    Sentences = laod_sent(Path1 + 'sentences.txt')

    SentimentWords = load_words(Path1 + 'sentimentwords.txt')
    DenyWords = load_words(Path1 + 'denywords.txt')
    DegreeWords = load_words(Path1 + 'degreewords.txt')
    StopWords = stop_words(Path1 + 'stopwords.txt')
    jieba.load_userdict(Path1 + 'userdict.txt')

    SegWords = seg_words(u'跌停板', StopWords)
    print (SegWords)
    SentimentWord = [(Word, SentimentWords[Word]) for Word in SegWords if Word in SentimentWords.keys()]
    print (SentimentWord)

    for item in Sentences.items():
        ind = item[0]
        Sentence = item[1]
        print (ind)
        Score = 0

        Count = 0
        for SegSentence in seg_sentences(Sentence, pouncts()):
            InnerScore = 0
            SegWords = seg_words(SegSentence, StopWords)
            SentimentWord = [(Word, SentimentWords[Word]) for Word in SegWords if Word in SentimentWords.keys()]
            Count += len(SentimentWord)

            # with open(Path1 + 'freq.txt', 'a') as f4:
            # f4.write(str(len(SentimentWord)) + '\n')

            for Tuple in SentimentWord:
                InnerScore += Tuple[1]

            for Word in SegWords:
                if Word in DenyWords.keys():
                    InnerScore *= DenyWords[Word]

                if Word in DegreeWords.keys():
                    InnerScore *= DegreeWords[Word]

            Score += InnerScore

        Score *= (2 ** len(re.findall(u'[!|！]', Sentence)))

        with open(Path1 + 'result.txt', 'a') as f2:

            f2.write(ind + ' ' + str(Score) + '\n')

        if Count == 0:
            with open(Path1 + 'count.txt', 'a') as f3:
                f3.write(ind + '\n')
