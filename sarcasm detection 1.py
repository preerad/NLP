#import numpy as np
#data_dic = {'mike':'Best Friend','superman': 'Henry Cavill', 'batman' : 'CBale'}
#print (data_dic.get('mike'))

import urllib
from   urllib.request import urlopen
import json
import numpy as np

url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json"
response = urllib.request.urlopen(url)
data = json.loads(response.read())

print ('I love you Miku ' +str(data[1]))

sentence = []
labels   = []
url      = []

for item in data:
    sentence.append(item['headline'])
    labels.append(item['is_sarcastic'])
    url.append(item['article_link'])

print (len(sentence))

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

train_senten = sentence[:20000]
train_label  = labels[:20000]
test_senten  = sentence [20000:]
test_label   = labels[20000:]

tokenize = Tokenizer(num_words = 100000 , oov_token='mike my love')
tokenize.fit_on_texts(train_senten)
word_index = tokenize.word_index
sentence_train = tokenize.texts_to_sequences(train_senten)
padded_train = pad_sequences(sentence_train , padding='post',maxlen=max_length,truncating=trunc_type)
print (list(word_index.items())[:10])


sentence_test = tokenize.texts_to_sequences(test_senten)
padded_test   = pad_sequences(sentence_test,padding='post',maxlen=max_length,truncating=trunc_type)

sentence_train_final  = np.array(sentence_train)
padded_train_final    = np.array(padded_train)
sentence_test_final   = np.array(sentence_test)
padded_test_final     = np.array(padded_test)

model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
                             tf.keras.layers.GlobalAveragePooling1D(),
                             tf.keras.layers.Dense(24,activation='relu'),
                             tf.keras.layers.Dense(1 , activation='sigmoid')])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

epoch_iter = 20
history = model.fit(sentence_train_final,padded_train_final,epochs=epoch_iter,validation_data=(sentence_test_final,padded_test_final))




