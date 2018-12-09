import nltk
from nltk.tokenize import word_tokenize
import tensorflow
import pandas

def calc_sample_words_rate(file):
    csv = pandas.read_csv(file).values

    words_count = 0
    sample_count = 0

    for col in csv:
        # print (col)
        for row in col:
            if isinstance(row, str):
                try:
                    # print (row)
                    tokens = word_tokenize(row)
                    sample_count += 1
                    words_count += len(tokens)
                    # print (tokens)
                except Exception as ex:
                    print (ex)

    print ( sample_count/(words_count/sample_count) )

if __name__ == "__main__":

    # nltk.download('punkt')
    # nltk.download('popular')

    calc_sample_words_rate("data/sample.csv")
