# from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.preprocessing import text, sequence
from tensorflow.python.keras import utils
from sklearn.preprocessing import LabelEncoder
import numpy as np

def train_and_predict(X_train, X_test, Y_train, Y_test):
    max_words = 1226
    tokenize = text.Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(X_train)

    x_train = tokenize.texts_to_matrix(X_train)
    x_test = tokenize.texts_to_matrix(X_test)

    encoder = LabelEncoder()
    encoder.fit(Y_train)
    y_train = encoder.transform(Y_train)
    y_test = encoder.transform(Y_test)

    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    # Parameters
    learning_rate = 0.02
    training_epochs = 10000

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, max_words])
    y = tf.placeholder(tf.float32, [None, num_classes])

    # Set model weights
    W = tf.Variable(tf.zeros([max_words, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b)
    # pred = tf.nn.sigmoid(tf.matmul(x, W) + b)

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: x_train, y: y_train})

            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))