import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

def train_and_predict(x_text, X_train, X_test, Y_train, Y_test):
    max_document_length = max([len(x.split(" ")) for x in x_text])

    vocab_processor = learn.preprocessing.VocabularyProcessor()

    x_train = np.array(list(vocab_processor.fit_transform(X_train)))
    x_test = np.array(list(vocab_processor.transform(X_test)))
    n_words = len(vocab_processor.vocabulary_)


def bag_of_words_model(features, target):
  """A bag-of-words model. Note it disregards the word order in the text."""
  target = tf.one_hot(target, 15, 1, 0)
  features = tf.contrib.layers.bow_encoder(
      features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
  logits = tf.contrib.layers.fully_connected(features, 15,
      activation_fn=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(),
      optimizer='Adam', learning_rate=0.01)
  return (
      {'class': tf.argmax(logits, 1),
       'prob': tf.nn.softmax(logits)},
      loss, train_op)