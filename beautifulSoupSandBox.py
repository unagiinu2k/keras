import requests
import bs4
#http://www.yoheim.net/blog.php?q=20160301
r2 = requests.get("http://cinema.intercritique.com/comment.cgi?new=1&page=1")
soup2 = bs4.BeautifulSoup(r2.content)
gdata2 = soup2.find_all('td',{'class':'title'})

for item2 in gdata2:
    #http://stackoverflow.com/questions/14257717/python-beautifulsoup-wildcard-attribute-id-search
    rating2 = item2.find_all('span',{'class': lambda L : L.startswith('rating')})[0].text[1]
    title2 = item2.find_all('a')[0].text
    urlHead2 = 'http://cinema.intercritique.com/'
    for link2 in item2('a'):
        try:
            lyriclink = urlHead2+link2.get('href')
            req = requests.get(lyriclink)
            lyricsoup = BeautifulSoup(req.content)
            lyricdata = lyricsoup.find_all('div',{'id':re.compile('lyric_space|lyrics')})[0].text
            eminemLyrics.append([title,lyricdata])
            print(title)
            print(lyricdata)
            print()
        except:
            pass



r3 = requests.get("http://www.lyrics.com")
soup3 = bs4.BeautifulSoup(r3.content)
#attribute3 = {'valign':'top' ,  'width':"54" ,  "align" :"center"}
attribute3 = {'width' : '100'}
gdata3 = soup3.find_all('td' ,attribute3)
urlHead3 = "http://www.lyrics.com"
runLyrics = []
for item3 in gdata3:
    link = item3("a")
        #.find_all("a" , {"style":"font-weight:bold;"})
    lyriclink = urlHead3 + link[0].get('href')
    req = requests.get(lyriclink)
    lyricsoup = BeautifulSoup(req.content)
    lyricdata = lyricsoup.find_all('div',{'id':re.compile('lyric_space|lyrics')})[0].text
    title = item3("a")[0].text
    runLyrics.append([title,lyricdata])


type(tmp)