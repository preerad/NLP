import urllib
from urllib.request import urlopen, Request
import numpy as np

url = 'https://gtsnjzec.coursera-apps.org/files/Week%201/Dinosaur%20Island%20--%20Character-level%20language%20model/dinos.txt'

#urllib.request.urlretrieve(url,'C:/Users/preer/Desktop/Dino.txt')

with open('C:/Users/preer/Desktop/Dino.txt','rb') as dino:
    html = dino.read()

html = html.lower()


print (html)

chars = list(set(html))

print(len(chars))



