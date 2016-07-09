#http://stackoverflow.com/questions/24873773/web-scraping-rap-lyrics-on-rap-genius-w-python

import re
import requests
import nltk
from bs4 import BeautifulSoup

url = 'http://www.lyrics.com/eminem'
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




Titles = [i[0] for i in eminemLyrics]
Lyrics = [i[1] for i in eminemLyrics]

from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer(min_df = 1)
X = Vectorizer.fit_transform(Lyrics)
X.toarray()
Vectorizer.get_feature_names()