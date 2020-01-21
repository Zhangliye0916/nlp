# encoding=utf-8


from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2016-10-31/', lang='zh')  # 英文使用 lang='en'

sentence = "清华大学位于北京。"

# sentence = "The book is very interesting."


# print nlp.word_tokenize(sentence)

# print nlp.pos_tag(sentence)

print nlp.parse(sentence)

print nlp.dependency_parse(sentence)
