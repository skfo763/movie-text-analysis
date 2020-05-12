from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import json
import os

from numpy import argmax
from tensorflow.compat.v2.keras.models import load_model
from tensorflow.compat.v2.keras.models import model_from_json


def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


def load_model():
    if os.path.isfile('train_docs.json'):
        json_file = open('movie-review-model.json', 'r')
        model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(model_json)
        model.load_weights('model.h5')
        return model
    else:
        raise ValueError('there is no file')


def predict_postitive_rate(review):
    model = load_model()
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)

    score = float(model.predict(data))
    if (score > 0.5):
        print("[{}]는 {:.2f}% 확률로 긍정적 리뷰입니다.\n".format(review, score * 100))
    else:
        print("[{}]는 {:.2f}% 확률로 부정적 리뷰입니다.\n".format(review, (1 - score) * 100))


predict_postitive_rate("인생 최고의 영화!")
predict_postitive_rate("최고의 배우. 그러나 답답한 스토리 전개. 배우가 아깝다.")