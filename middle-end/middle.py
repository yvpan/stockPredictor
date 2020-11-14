import urllib
import requests
import codecs
from bs4 import BeautifulSoup as bs

infile = open("../front-end/frontOut.txt")
lines = infile.readlines()
symbol = lines[0].split("\t")[0]
infile.close()

URL = "https://www.google.com/finance/quote/" + str(symbol) + ":NASDAQ"
resp = requests.get(URL)
soup = bs(resp.text, "html.parser")
articles = soup.find_all("div", attrs = {"class": "AoCdqe"})
outfile = codecs.open('./middleOut.txt', 'w',"utf-8")
for e in articles:
    title = e.text
    title = title.replace("\n", " ")
    outfile.write(title + '\n')
outfile.close()
