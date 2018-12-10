from data_loader import load_data
import naive_bayes
import linear_svm
import logistic_regression
import word2vec
import word2vec_logistic_regression
import ml_relu_softmax
import tf_logistic_regression

if __name__ == "__main__":
    path = 'data/sample.csv'
    (X_train, X_test, Y_train, Y_test, tags) = load_data(path)

    # naive_bayes.train_and_predict(X_train, X_test, y_train, y_test, tags)
    # linear_svm.train_and_predict(X_train, X_test, y_train, y_test, tags)
    # logistic_regression.train_and_predict(X_train, X_test, y_train, y_test, tags)
    #

    # (X_train_word_average, X_test_word_average) = word2vec.word2vec_data('.'.join(X_train), '.'.join(X_test))
    # word2vec_logistic_regression.train_and_predict(X_train_word_average, X_test_word_average, Y_train, Y_test, tags)

    ml_relu_softmax.train_and_predict(X_train, X_test, Y_train, Y_test)

    # tf_logistic_regression.train_and_predict(X_train, X_test, Y_train, Y_test)