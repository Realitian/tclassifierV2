import nltk
from nltk.tokenize import word_tokenize
import tensorflow
import pandas
import re
from nltk.corpus import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    return text

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
                    tokens = word_tokenize(clean_text(row))
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
