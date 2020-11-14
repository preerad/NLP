def process_tweet(tweet):
import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random
print ('I love my miku')
nltk.download('twitter_samples')
positive_all = twitter_samples.strings('positive_tweets.json')
negative_all = twitter_samples.strings('negative_tweets.json')
#fig = plt.figure(figsize=(5,5))
#labels = 'Positive' , 'Negative'
#sizes = [len(positive_all[1:1000]),len(negative_all)]
#plt.pie(sizes,labels=labels,shadow=True,startangle=90,autopct='%1.1f%%')
#plt.axis('equal')
#plt.show()
print ('\033[36m' + positive_all[random.randint(0,5000)])
print ('\033[31m' + negative_all[random.randint(0,5000)])
nltk.download('stopwords')
import re
import string

from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer

tweet = positive_all[2277]
print('preetha' +str(tweet))

tweet = re.sub(r'#','',tweet)

print('mike' +str(tweet))

tweet = re.sub(r',',' ',tweet)

tweet = tweet.lower()

print('mike baby' +str(tweet))

#tw1 = tweet.split()
#print(tw1)

tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)

tweet_token = tokenizer.tokenize(tweet)
print (len(tweet_token))

w1 = stopwords.words('english')

print ('stop words',w1)
punc = string.punctuation
print ('punc' , string.punctuation)

tweet_clean = []

for word in tweet_token:
    if (word not in w1 and
       word not in punc):
       tweet_clean.append(word)

       print (tweet_clean)

       stemmer = PorterStemmer()

       tweet_stem = []

       for word in tweet_clean:
           stem_word = stemmer.stem(word)
           tweet_stem.append(word)

           print ('stemmed tweet',tweet_stem)

       return tweet_stem

import numpy as np

ys = np.append(np.ones(len(positive_all)),np.zeros(len(negative_all)))

build_freq(tweet_stem, ys)



    ##### build frequencies

def build_freq(tweets,ys):

  ylist = np.squeeze(ys).tolist()

  freq = {}

  for y,tweets in zip(ys,tweets):
    for word in process_tweet(tweet):
        pair = (word,y)
        if pair in freq:
            freq[pair] += 1
        else:
            freq[pair]  = 1

    return freq


           














