from data_loader import load_data
import naive_bayes
import linear_svm
import logistic_regression

if __name__ == "__main__":
    path = 'data/sample.csv'
    (X_train, X_test, y_train, y_test, tags) = load_data(path)
    naive_bayes.train_and_predict(X_train, X_test, y_train, y_test, tags)
    linear_svm.train_and_predict(X_train, X_test, y_train, y_test, tags)
    logistic_regression.train_and_predict(X_train, X_test, y_train, y_test, tags)