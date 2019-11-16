# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
import gc
from bs4 import BeautifulSoup

import sys
import os

# os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import keras.backend
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate, Dropout
from keras.models import Model
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
    data_train = pd.read_csv('./labeledTrainData.tsv', sep='\t')
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

    print('Number of positive and negative reviews in traing and validation set ')
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

    print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    with open('./models/conv/embedding_matrix.pkl', 'wb')as f:
        pickle.dump(embedding_matrix, f)

    with open('./models/conv/embeddings_index.pkl', 'wb')as f:
        pickle.dump(embeddings_index, f)

    with open('./models/conv/tokenizer.pkl', 'wb')as f:
        pickle.dump(tokenizer, f)

    with open('./models/conv/save_vec_all.pkl', 'wb')as f:
        pickle.dump([embedding_matrix, embeddings_index, tokenizer, x_train, y_train, x_val, y_val], f)

    return embedding_matrix, embeddings_index, tokenizer, x_train, y_train, x_val, y_val

def train():
    """简单的卷积神经网络"""
    embedding_matrix, embeddings_index, tokenizer, x_train, y_train, x_val, y_val = save_vec_et()

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
            l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
            l_pool1 = MaxPooling1D(5)(l_cov1)
            l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
            l_pool2 = MaxPooling1D(5)(l_cov2)
            l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
            l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
            l_flat = Flatten()(l_pool3)
            l_dense = Dense(128, activation='relu')(l_flat)
            preds = Dense(2, activation='softmax')(l_dense)

            model = Model(sequence_input, preds)
            model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['acc'])

            print("model fitting - simplified convolutional neural network")
            model.summary()
            model.fit(x_train, y_train, validation_data=(x_val, y_val),
                      epochs=10, batch_size=128)

            model.save('./models/model_conv_1.h5')

def train2():
    """
    复杂点的卷积神经网络
    :return:
    """
    with open('./models/conv/save_vec_all.pkl', 'rb')as f:
        embedding_matrix, embeddings_index, tokenizer, x_train, y_train, x_val, y_val = pickle.load(f)

    word_index = tokenizer.word_index

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    # applying a more complex convolutional approach
    convs = []
    filter_sizes = [3,4,5]

    graph = tf.Graph()  # Tensorflow graph
    with graph.as_default():
        session = tf.Session()  # Tensorflow session
        with session.as_default():
            sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)

            for fsz in filter_sizes:
                l_conv = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_sequences)
                l_pool = MaxPooling1D(5)(l_conv)
                convs.append(l_pool)

            # l_merge = Merge(mode='concat', concat_axis=1)(convs)
            l_merge = concatenate(convs, axis=1)
            l_cov1= Conv1D(128, 5, activation='relu')(l_merge)
            l_pool1 = MaxPooling1D(5)(l_cov1)
            l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
            l_pool2 = MaxPooling1D(30)(l_cov2)
            l_flat = Flatten()(l_pool2)
            l_dense = Dense(128, activation='relu')(l_flat)
            preds = Dense(2, activation='softmax')(l_dense)

            model = Model(sequence_input, preds)
            model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['acc'])

            print("model fitting - more complex convolutional neural network")
            model.summary()
            model.fit(x_train, y_train, validation_data=(x_val, y_val),
                      epochs=20, batch_size=50)
            model.save('./models/model_conv_2.h5')

def predict(text):
    with open('./models/conv/save_vec_all.pkl', 'rb')as f:
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
            l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
            l_pool1 = MaxPooling1D(5)(l_cov1)
            l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
            l_pool2 = MaxPooling1D(5)(l_cov2)
            l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
            l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
            l_flat = Flatten()(l_pool3)
            l_dense = Dense(128, activation='relu')(l_flat)
            preds = Dense(2, activation='softmax')(l_dense)

            model = Model(sequence_input, preds)
            model.load_weights('./models/model_conv_1.h5', by_name=True, reshape=True)

            sequences = tokenizer.texts_to_sequences([text])

            X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

            ret = model.predict(X)
            print(ret)
    # keras.backend.clear_session()

def predict2(text):
    with open('./models/conv/save_vec_all.pkl', 'rb')as f:
        embedding_matrix, embeddings_index, tokenizer, x_train, y_train, x_val, y_val = pickle.load(f)

    word_index = tokenizer.word_index

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    # applying a more complex convolutional approach
    convs = []
    filter_sizes = [3,4,5]

    graph = tf.Graph()  # Tensorflow graph
    with graph.as_default():
        session = tf.Session()  # Tensorflow session
        with session.as_default():
            sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)

            for fsz in filter_sizes:
                l_conv = Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)
                l_pool = MaxPooling1D(5)(l_conv)
                convs.append(l_pool)

            # l_merge = Merge(mode='concat', concat_axis=1)(convs)
            l_merge = concatenate(convs, axis=1)
            l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
            l_pool1 = MaxPooling1D(5)(l_cov1)
            l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
            l_pool2 = MaxPooling1D(30)(l_cov2)
            l_flat = Flatten()(l_pool2)
            l_dense = Dense(128, activation='relu')(l_flat)
            preds = Dense(2, activation='softmax')(l_dense)

            model = Model(sequence_input, preds)
            model.load_weights('./models/model_conv_2.h5', by_name=True, reshape=True)

            sequences = tokenizer.texts_to_sequences([text])

            X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            ret = model.predict(X)
            print(ret)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
        train2()
    else:
        text = """With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."""
        text = """This movie could have been very good, but comes up way short. Cheesy special effects and so-so acting. I could have looked past that if the story wasn't so lousy. If there was more of a background story, it would have been better. The plot centers around an evil Druid witch who is linked to this woman who gets migraines. The movie drags on and on and never clearly explains anything, it just keeps plodding on. Christopher Walken has a part, but it is completely senseless, as is most of the movie. This movie had potential, but it looks like some really bad made for TV movie. I would avoid this movie."""
        predict(text)
        # gc.collect()  # 手动清理内存
        predict2(text)

if __name__ == '__main__':
    main()


# 卷积神经网络对电影评论二分类
# https://github.com/richliao/textClassifier
