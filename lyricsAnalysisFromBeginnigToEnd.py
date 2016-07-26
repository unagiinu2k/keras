import requests
import bs4
import re
#http://www.yoheim.net/blog.php?q=20160301


import numpy
runLyrics = []
urlHead3 = "http://www.lyrics.com"
attribute3 = {'width' : '100'}
rankRanges = numpy.arange(0,210,30)
for r in rankRanges:
    runUrl = "http://www.lyrics.com/tophits/home_countries/" + str(r) + "/US"
    r3 = requests.get(runUrl)
    soup3 = bs4.BeautifulSoup(r3.content)
    gdata3 = soup3.find_all('td' ,attribute3)
    for item3 in gdata3:
        link = item3("a")
            #.find_all("a" , {"style":"font-weight:bold;"})
        lyriclink = urlHead3 + link[0].get('href')
        req = requests.get(lyriclink)
        lyricsoup = bs4.BeautifulSoup(req.content)
        lyricdata = lyricsoup.find_all('div',{'id':re.compile('lyric_space|lyrics')})[0].text
        title = item3("a")[0].text
        runLyrics.append([title,lyricdata])


Lyrics = [s[1] for s in runLyrics]

vocab = [s.split() for s in Lyrics]

import gensim


checkVariables = False

#単語レベルに分解したlistのlistを食わせるのが正しい。toy exampleではmin_countを1とかにしておくのが良い
word2vecModel = gensim.models.Word2Vec(vocab , min_count=1 , size = 20 ,  iter=100)

runVocab =word2vecModel.vocab
if checkVariables:
    runVocab.keys()

word2vecModel.most_similar(positive=[u"I"])
word2vecModel.most_similar(positive=[u"dreamer"])
word2vecModel.most_similar(positive=[u"nightmare"])

word2vecModel.most_similar(positive=[u"love"])
word2vecModel.most_similar(positive=[u"Let"])
word2vecModel.most_similar(positive=[u"In"])
word2vecModel.most_similar(positive=[u"tell" , "me"])



#LDA
#http://sucrose.hatenablog.com/entry/2013/10/29/001041
Dictionary = gensim.corpora.Dictionary(vocab)
if checkVariables:
    print(Dictionary)
    tmp = Dictionary.values()
    vocab[0]
    Dictionary.doc2bow(vocab[0])
    Dictionary.get(0)
    Dictionary.get(1)
    Dictionary.doc2bow(["you" , "you" , "come"])

Corpus = [Dictionary.doc2bow(t) for t in vocab]
#https://radimrehurek.com/gensim/tut2.html
Tfidf = gensim.models.TfidfModel(Corpus , id2word=Dictionary)
corpusTfidf = Tfidf[Corpus]
useTfidfCorpus = True
if useTfidfCorpus:
    Lda = gensim.models.LdaModel(corpus=corpusTfidf , id2word = Dictionary , num_topics=20)
else:
    Lda = gensim.models.LdaModel(corpus=Corpus , id2word = Dictionary , num_topics=20)

if checkVariables:
    tmp = Lda.show_topics(-1) #show_topicsとshow_topicで挙動が違うのに注意！
    [t for t in Lda.show_topic(-1)]

for t in Lda.show_topic(-1):
    print(t)
for t in Lda[Corpus]:
    print(t)

Lda.show_topic(-1)[1]
lda[Corpus[1]]
gensim.corpora.LowCorpus(vocab)


from __future__ import print_function
warned_of_error = False

def create_cloud(oname, words,maxsize=120, fontname='Lobster'):
    '''Creates a word cloud (when pytagcloud is installed)

    Parameters
    ----------
    oname : output filename
    words : list of (value,str)
    maxsize : int, optional
        Size of maximum word. The best setting for this parameter will often
        require some manual tuning for each input.
    fontname : str, optional
        Font to use.
    '''
    try:
        from pytagcloud import create_tag_image, make_tags
    except ImportError:
        if not warned_of_error:
            print("Could not import pytagcloud. Skipping cloud generation")
        return

    # gensim returns a weight between 0 and 1 for each word, while pytagcloudh
    # expects an integer word count. So, we multiply by a large number and
    # round. For a visualization this is an adequate approximation.
    # We also need to flip the order as gensim returns (value, word), whilst
    # pytagcloud expects (word, value):
    words = [(v,int(w*10000)) for v,w in words]
    tags = make_tags(words, maxsize=maxsize)
    create_tag_image(tags, oname, size=(1800, 1200), fontname=fontname)


create_cloud('Lyrics.png', Lda.show_topic(-1), maxsize = 50, fontname='Cardo')