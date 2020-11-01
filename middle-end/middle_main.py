import urllib
import requests
import codecs
from bs4 import BeautifulSoup as bs
companys = []

infile = open("result.txt")
lines = infile.readlines()

for i in range(lines.__len__()):
    store = []
    for word in lines[i].split():
        word = word.strip(" ")
        store.append(word)
    companys.append(store)
infile.close()

for company in companys:

    query = company[1]
    URL = "https://www.forbes.com/search/?q={query}&startdate=month&sort=score"

    resp = requests.get(URL)
    soup = bs(resp.text, 'html.parser')

    articles = soup.find_all(
        'a', attrs = {"class": 'stream-item__title'}
    )

    outfile = codecs.open('crawlNews.txt', 'w',"utf-8")
    for e in articles:
        title = e.text
        outfile.write(title + '\n')
    outfile.close()
