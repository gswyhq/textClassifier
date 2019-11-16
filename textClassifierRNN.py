# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

# os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers as initializations

import tensorflow as tf

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()

def save_vec_et():
    # https://github.com/charlesliucn/kaggle-in-python/blob/master/kaggle_competitions/IMDB/labeledTrainData.tsv
    data_train = pd.read_csv('labeledTrainData.tsv', sep='\t')
    print(data_train.shape)

    texts = []
    labels = []

    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx], features="lxml")
        texts.append(clean_str(text.get_text()))
        labels.append(data_train.sentiment[idx])


    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print('Traing and validation set number of positive and negative reviews')
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    GLOVE_DIR = "."
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    with open('./models/rnn/embedding_matrix.pkl', 'wb')as f:
        pickle.dump(embedding_matrix, f)

    with open('./models/rnn/embeddings_index.pkl', 'wb')as f:
        pickle.dump(embeddings_index, f)

    with open('./models/rnn/tokenizer.pkl', 'wb')as f:
        pickle.dump(tokenizer, f)

    with open('./models/rnn/save_vec_all.pkl', 'wb')as f:
        pickle.dump([embedding_matrix, embeddings_index, tokenizer, x_train, y_train, x_val, y_val], f)

    return tokenizer, embedding_matrix, embeddings_index, x_train, y_train, x_val, y_val

def train_bi_LSTM():
    tokenizer, embedding_matrix, embeddings_index, x_train, y_train, x_val, y_val = save_vec_et()
    word_index = tokenizer.word_index

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    graph = tf.Graph()  # Tensorflow graph
    with graph.as_default():
        session = tf.Session()  # Tensorflow session
        with session.as_default():
            sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)
            l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
            preds = Dense(2, activation='softmax')(l_lstm)
            model = Model(sequence_input, preds)
            model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['acc'])

            print("model fitting - Bidirectional LSTM")
            model.summary()
            model.fit(x_train, y_train, validation_data=(x_val, y_val),
                      nb_epoch=10, batch_size=50)
            model.save('./models/model_bi_lstm.h5')

# Attention GRU network
class AttentionLayer(Layer):
    # https://github.com/ttjjlw/text_classify_by_keras/blob/master/Models/Bi_gru_att.py
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

def train_attention_GRU():
    with open('./models/rnn/save_vec_all.pkl', 'rb')as f:
        embedding_matrix, embeddings_index, tokenizer, x_train, y_train, x_val, y_val = pickle.load(f)
    word_index = tokenizer.word_index

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)


    graph = tf.Graph()  # Tensorflow graph
    with graph.as_default():
        session = tf.Session()  # Tensorflow session
        with session.as_default():
            sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)
            l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
            l_att = AttentionLayer()(l_gru)
            preds = Dense(2, activation='softmax')(l_att)
            model = Model(sequence_input, preds)
            model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['acc'])

            print("model fitting - attention GRU network")
            model.summary()

            model.fit(x_train, y_train, validation_data=(x_val, y_val),
                      nb_epoch=10, batch_size=50)
            model.save('./models/model_att_gru.h5')

def predict_bi_LSTM(text):
    with open('./models/rnn/save_vec_all.pkl', 'rb')as f:
        embedding_matrix, embeddings_index, tokenizer, x_train, y_train, x_val, y_val = pickle.load(f)
    word_index = tokenizer.word_index

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    graph = tf.Graph()  # Tensorflow graph
    with graph.as_default():
        session = tf.Session()  # Tensorflow session
        with session.as_default():
            sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)
            l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
            preds = Dense(2, activation='softmax')(l_lstm)
            model = Model(sequence_input, preds)

            model.load_weights('./models/model_bi_lstm.h5', by_name=True, reshape=True)
            sequences = tokenizer.texts_to_sequences([text])

            X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            ret = model.predict(X)
            print(ret)

def predict_attention_GRU(text):
    with open('./models/rnn/save_vec_all.pkl', 'rb')as f:
        embedding_matrix, embeddings_index, tokenizer, x_train, y_train, x_val, y_val = pickle.load(f)
    word_index = tokenizer.word_index

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)


    graph = tf.Graph()  # Tensorflow graph
    with graph.as_default():
        session = tf.Session()  # Tensorflow session
        with session.as_default():
            sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)
            l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
            l_att = AttentionLayer()(l_gru)
            preds = Dense(2, activation='softmax')(l_att)
            model = Model(sequence_input, preds)
            model.load_weights('./models/model_att_gru.h5', by_name=True, reshape=True)
            sequences = tokenizer.texts_to_sequences([text])

            X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            ret = model.predict(X)
            print(ret)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_bi_LSTM()
        train_attention_GRU()
    else:
        text = """With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."""
        text = """This movie could have been very good, but comes up way short. Cheesy special effects and so-so acting. I could have looked past that if the story wasn't so lousy. If there was more of a background story, it would have been better. The plot centers around an evil Druid witch who is linked to this woman who gets migraines. The movie drags on and on and never clearly explains anything, it just keeps plodding on. Christopher Walken has a part, but it is completely senseless, as is most of the movie. This movie had potential, but it looks like some really bad made for TV movie. I would avoid this movie."""
        predict_bi_LSTM(text)
        # gc.collect()  # 手动清理内存
        predict_attention_GRU(text)

if __name__ == '__main__':
    main()


# 双向循环神经网络对电影评论二分类
# https://github.com/richliao/textClassifier

