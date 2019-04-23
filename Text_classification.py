import pandas as pd

df = pd.read_pickle("Consumer_complaints.pkl.zip")

df = df[["Consumer complaint narrative", "Product"]]
df = df.dropna()

df = df[:10000]

#df.shape

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df["Product"] = le.fit_transform(df['Product'])


df.Product.value_counts()

df = df[(df["Product"]==6)|(df["Product"]==7)|(df["Product"]==10)|(df["Product"]==4)|(df["Product"]==5)|(df["Product"]==15)]
#df.shape
df.Product.value_counts()


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english")

vector = cv.fit_transform(df["Consumer complaint narrative"])

X = vector.toarray()
y = df["Product"]

#X.shape
#y.shape

from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(X,y,random_state = 42,test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, max_depth=4)

rfc.fit(X_train,y_train)
rfc.score(X_test,y_test)

from sklearn.metrics import classification_report, accuracy_score

y_pred = rfc.predict(X_test)
print (classification_report(y_test,y_pred))

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)
print (accuracy_score(y_test,y_pred))
print (classification_report(y_test,y_pred))

