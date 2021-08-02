
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Embed-Encode-Attend-Predict
import pandas as pd
import tensorflow as tf
import numpy as np
import time
import os, sys
from sklearn import metrics
from visualize_attention import attentionDisplay
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

# x_train, y_train, x_test, y_test, n_words = process_inputs(vocab_processor, wiki)

# STEP1: Embed
# Random initialization using embed_sequence
def embed(features):
    n_words = len(vocab_processor.vocabulary_)
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

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell,
                                                rnn_bw_cell,
                                                word_vectors,
                                                dtype=tf.float32,
                                                time_major=False)
    return outputs

# STEP3: Attend
def attend(inputs, attention_size, attention_depth):
    inputs = tf.concat(inputs, axis=2)

    inputs_shape = inputs.shape
    sequence_length = inputs_shape[1].value
    final_layer_size = inputs_shape[2].value

    x = tf.reshape(inputs, [-1, final_layer_size])
    for _ in range(attention_depth - 1):
        x = tf.layers.dense(x, attention_size, activation=tf.nn.relu)
    x = tf.layers.dense(x, 1, activation=None)
    logits = tf.reshape(x, [-1, sequence_length, 1])
    alphas = tf.nn.softmax(logits, dim=1)

    output = tf.reduce_sum(inputs * alphas, 1)

    return output, alphas

# STEP4: Predict
def estimator_spec_for_softmax_classification(logits, labels, mode, alphas):
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits),
                'attention': alphas
            })
    onehot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes
        ),
        'auc': tf.metrics.auc(
            labels=labels, predictions=predicted_classes
        )
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def predict(encoding, labels, mode, alphas):
    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
    return estimator_spec_for_softmax_classification(logits=logits, labels=labels, mode=mode, alphas=alphas)


# STEP5: Complete Model Architecture
# Embed -> Encode -> Attend -> Predict
def bi_rnn_model(features, labels, mode):
    word_vectors = embed(features)
    outputs = encode(word_vectors)
    encoding, alphas = attend(outputs,
                              hparams['attention_size'],
                              hparams['attention_depth'])
    return predict(encoding, labels, mode, alphas)


def main():
    # Download the data from Figshare and cleansing and splitting it for use in training
    # download_figshare()
    # process_figshare()
    x_train, y_train, x_test, y_test, n_words = process_inputs(vocab_processor, wiki)

    # # STEP5: Train
    # current_time = str(int(time.time()))
    # model_dir = os.path.join('checkpoints', current_time)
    classifier = tf.estimator.Estimator(model_fn=bi_rnn_model,
                                        model_dir=model_dir)

    # # TRAIN
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={WORDS_FEATURE: x_train},
    #     y=y_train,
    #     batch_size=hparams['batch_size'],
    #     num_epochs=None,
    #     shuffle=True)

    classifier.train(input_fn=train_input_fn,
                    steps=NUM_STEPS)

    # STEP6: Predict and Evaluate Model
    # PREDICT
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)

    y_predicted = []
    alphas_predicted = []
    for p in predictions:
        y_predicted.append(p['class'])
        alphas_predicted.append(p['attention'])
    # Evaluate
    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy: {0:f}'.format(scores['accuracy']))
    print('AUC: {0:f}'.format(scores['auc']))

    display = attentionDisplay(vocab_processor, classifier)
    display.display_prediction_attention('Thanks for your help editing this.')
    display.display_prediction_attention('God damn it!')
    display.display_prediction_attention('Gosh darn it!')
if __name__ == "__main__":
    sys.exit(main())
