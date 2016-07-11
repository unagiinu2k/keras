#http://stackoverflow.com/questions/24873773/web-scraping-rap-lyrics-on-rap-genius-w-python

import re
import requests
import nltk
from bs4 import BeautifulSoup

#url = 'http://www.lyrics.com/eminem'
url = 'http://www.lyrics.com/radiohead'
r = requests.get(url)
soup = BeautifulSoup(r.content)
gdata = soup.find_all('div',{'class':'row'})

eminemLyrics = []

for item in gdata:
    title = item.find_all('a',{'itemprop':'name'})[0].text
    lyricsdotcom = 'http://www.lyrics.com'
    for link in item('a'):
        try:
            lyriclink = lyricsdotcom+link.get('href')
            req = requests.get(lyriclink)
            lyricsoup = BeautifulSoup(req.content)
            lyricdata = lyricsoup.find_all('div',{'id':re.compile('lyric_space|lyrics')})[0].text
            eminemLyrics.append([title,lyricdata])
            print(title)
            print(lyricdata)
            print()
        except:
            pass




removeKey = '\n\nSubmit LyricsYour name will be printed as part of the credit when your lyric is approved. \n'
runRows = [i[1] == removeKey for i in eminemLyrics]
eminemLyrics[[runRows]]
Titles = [i[0] for i in eminemLyrics if i[1] != removeKey]
Lyrics = [i[1] for i in eminemLyrics if i[1] != removeKey]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


import nltk.stem
englishStemmer = nltk.stem.SnowballStemmer('english')

class stemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (englishStemmer.stem(w) for w in analyzer(doc))

vectorizerType = "stem"
if vectorizerType == "simplest" :
    Vectorizer = CountVectorizer(min_df = 1)
elif vectorizerType "stem":
    # p64 of the book
    Vectorizer = stemmedTfidfVectorizer(min_df = 1 , stop_words= "english"   ,decode_error="ignore")


X = Vectorizer.fit_transform(Lyrics)
X.toarray()
featureNames = Vectorizer.get_feature_names()
len(featureNames)

#K-means
#http://qiita.com/ynakayama/items/1223b6844a1a044e2e3b
from sklearn.cluster import KMeans
kmeansModel = KMeans(n_clusters= 4 ,  random_state= 10).fit(X)
Labels = kmeansModel.labels_
import pandas
import numpy
Titles = numpy.asarray(Titles)
clusterResult = pandas.DataFrame({"Title" : Titles , "Label" : Labels})

runGroups = []
for i in set(clusterResult["Label"]):
    runGroups.append(clusterResult.query('Label == ' +  str(i)))

#word2vec
#http://tjo.hatenablog.com/entry/2014/06/19/233949#f-5d99effc

#word2vec
import gensim
if False:
    #http://stackoverflow.com/questions/33989826/python-gensim-runtimeerror-you-must-first-build-vocabulary-before-training-th
    tmp = gensim.models.Word2Vec([["nowhere",  "man",  "please" , "listen"] , ["the" , "long" ,  "and" , "winding" , "road"]] , size = 200 , min_count= 1)

#word2vecData = d2vec.Text8Corpus(Lyrics[5:10])
vocab = [s.split() for s in Lyrics]
word2vecModel = gensim.models.Word2Vec(vocab , size = 200 , iter=100)
word2vecModel.most_similar(positive=[u"song"])
word2vecModel.most_similar(positive=[u"tell" , "me"])
word2vecModel["song"]


#LDA
#http://sucrose.hatenablog.com/entry/2013/10/29/001041
Dictionary = gensim.corpora.Dictionary(vocab)
print(Dictionary)
Dictionary.doc2bow(vocab[1])
Corpus = [Dictionary.doc2bow(t) for t in vocab]
Token2id = Dictionary.token2id
lda = gensim.models.LdaModel(corpus=Corpus , id2word = Dictionary)
lda.show_topic(-1)[1]
gensim.corpora.LowCorpus(vocab)