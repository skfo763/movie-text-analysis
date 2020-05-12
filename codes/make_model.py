import json
import os
import nltk
import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint
from konlpy.tag import Okt
from matplotlib import font_manager, rc

from keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

%matplotlib inline


def read_data(filename):
    with open(filename, 'r') as buffer:
        data = [line.split('\t') for line in buffer.read().splitlines()]
        data = data[1:]
    return data


def tokenize(document):
    return ['/'.join(tag) for tag in okt.pos(document, norm=True, stem=True)]


def showGraph(data):
    font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)

    plt.figure(figsize=(20, 10))
    data.plot(50)


def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


okt = Okt()
train_data = read_data('ratings_train.txt')
test_data = read_data('ratings_test.txt')

if os.path.isfile('train_docs.json'):
    with open('train_docs.json') as fileBuffer:
        train_docs = json.load(fileBuffer)
    with open('test_docs.json') as fileBuffer:
        test_docs = json.load(fileBuffer)
else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]

    with open('train_docs.json', 'w', encoding="utf-8") as file:
        json.dump(train_docs, file, ensure_ascii=False, indent="\t")
    with open('test_docs.json', 'w', encoding="utf-8") as file:
        json.dump(test_docs, file, ensure_ascii=False, indent="\t")

tokens = [token for data in train_docs for token in data[0]]
text = nltk.Text(tokens, name='NMSC')
# showGraph(text)

selected_words = [f[0] for f in text.vocab().most_common(1000)]

train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]
train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]

x_train = np.asarray(train_x).astype('float32')
x_test = np.asarray(test_x).astype('float32')

y_train = np.asarray(train_y).astype('float32')
y_test = np.asarray(test_y).astype('float32')


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
             loss=losses.binary_crossentropy,
             metrics=[metrics.binary_accuracy])

model.fit(x_train, y_train, epochs=10, batch_size=512)
results = model.evaluate(x_test, y_test)

json_string = model.to_json()
with open("movie-review-model.json", "w") as json_file:
  json_file.write(json_string)
print("Saved model for json format")

model.save_weights("model.h5")
print("Saved weight for .h5 format")