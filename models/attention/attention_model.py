
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Embed-Encode-Attend-Predict
import pandas as pd
import tensorflow as tf
import numpy as np
import time
import os, sys
from sklearn import metrics
# from visualize_attention import attentionDisplay
from process_figshare import download_figshare, process_figshare
print(tf.__version__)

SPLITS = ['train', 'dev', 'test']

wiki = {}
for split in SPLITS:
    wiki[split] = pd.read_csv('data/wiki_%s.csv' % split)

print(wiki['train'].head())

# Hyperparameters
hparams = {'max_document_length': 60,
           'embedding_size': 50,
           'rnn_cell_size': 128,
           'batch_size': 256,
           'attention_size': 32,
           'attention_depth': 2}

MAX_LABEL = 2
WORDS_FEATURE = 'words'
NUM_STEPS = 300

# STEP0: Text Preprocessing
# Initialize the vocabulary processor
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(hparams['max_document_length'])

def process_inputs(vocab_processor, df, train_label='train', test_label='test'):
    # For simplicity, we call our features x and our oupus y
    x_train = df['train'].comment
    y_train = df['train'].is_toxic
    x_test = df['test'].comment
    y_test = df['test'].is_toxic

    # Train the vabac_processor from the training set
    x_train = vocab_processor.fit_transform(x_train)
    # Transform our test set with the vocabulary processor
    x_test = vocab_processor.transform(x_test)

    # we need these to be np.arrays instead of generators
    x_train = np.array(list(x_train))
    x_test = np.array(list(x_test))
    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    # Return the transformed data and the number of n_words
    return x_train, y_train, x_test, y_test, n_words


# STEP1: Embed
# Random initialization using embed_sequence
def embed(features):
    word_vectors = tf.contrib.layers.embed_sequence(
        features[WORDS_FEATURE],
        vocab_size=n_words,
        embed_dim=hparams['embedding_size'])
    return word_vectors

# STEP2: Encode
# Recurrent Neural Network (RNN) : useful encodign sequential information like sentences
def encode(word_vectors):
    rnn_fw_cell = tf.contrib.rnn.GRUCell(hparams['rnn_cell_size'])
    rnn_bw_cell = tf.contrib.rnn.GRUCell(hparams['rnn_cell_size'])

    ouputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell,
                                                rnn_bw_cell,
                                                word_vectors,
                                                dtype=tf.float32,
                                                time_major=False)
    return outputs


def main():
    # Download the data from Figshare and cleansing and splitting it for use in training
    # download_figshare()
    # process_figshare()
    x_train, y_train, x_test, y_test, n_words = process_inputs(vocab_processor, wiki)


if __name__ == "__main__":
    sys.exit(main())
