import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)

    X = []
    y = []

    tags = []

    for col in df:
        tags.append(col)

        for row in df[col]:

            if isinstance(row, str):
                X.append(row)
                y.append(col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

    return (X_train, X_test, y_train, y_test, tags)