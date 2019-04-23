import pandas as pd
df = pd.read_csv("new_complaints.csv")
df_copy = df.copy()
df = df[["Consumer complaint narrative", "Product"]]

df = df.dropna()

df.head(10)

sents = df["Consumer complaint narrative"].iloc[:50].tolist()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

words = [word_tokenize(x) for x in sents]

words = [item for sublist in words for item in sublist]
words = [x.lower() for x in words]
custom = stopwords.words("english")+list(punctuation)

print(custom)

words = [x for x in words if x not in custom]

print(words)

from collections import Counter

d = dict(Counter(words))

import operator

sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)

print(sorted_d)
