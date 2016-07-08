#http://stackoverflow.com/questions/24873773/web-scraping-rap-lyrics-on-rap-genius-w-python

from urlparse import urljoin
from bs4 import BeautifulSoup
import requests


BASE_URL = "http://genius.com"
artist_url = "http://genius.com/artists/Andre-3000/"

response = requests.get(artist_url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36'})

soup = BeautifulSoup(response.text, "lxml")
for song_link in soup.select('ul.song_list > li > a'):
    link = urljoin(BASE_URL, song_link['href'])
    response = requests.get(link)
    soup = BeautifulSoup(response.text)
    lyrics = soup.find('div', class_='lyrics').text.strip()

    # tokenize `lyrics` with nltk