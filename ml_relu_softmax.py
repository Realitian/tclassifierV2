import numpy as np
from sklearn.preprocessing import LabelEncoder

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.preprocessing import text, sequence
from tensorflow.python.keras import utils
from tensorflow.python.keras.callbacks import TensorBoard
import time

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

    batch_size = 32
    epochs = 1000

    # Build the model
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[tensorboard])

    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)
    print('Test accuracy:', score[1])