import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression
import string

np.random.seed(123456)
news_df = pd.read_csv("uci-news-aggregator.csv", sep = ",")
news_df['CATEGORY'] = news_df.CATEGORY.map({ 'b': 1, 't': 2, 'e': 3, 'm': 4 })
news_df['TITLE'] = news_df.TITLE.map(lambda x: x.lower().translate(str.maketrans('','', string.punctuation)))

vectorizer = CountVectorizer(stop_words='english')
x = vectorizer.fit_transform(news_df['TITLE'])

encoder = LabelEncoder()
y = encoder.fit_transform(news_df['CATEGORY'])

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,random_state=42)

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LinearRegression())
# Fit the classifier to the training data
clf.fit(x_train, y_train)
#clf.score(x_test,y_test)
y_pred=clf.predict(x_test)

tn,fn,tp,fp=confusion_matrix(y_test,y_pred)

print(confusion_matrix(y_test,y_pred))
print("===================================")
print(classification_report(y_test,y_pred))
print("===================================")
print("Accuracy score:",accuracy_score(y_test,y_pred))
print("===================================")
print(clf.predict(vectorizer.transform(["nescafe is a product from nestle"])))

plt.plot(tn)
plt.plot(tp)
plt.plot(fn)
plt.plot(fp)
plt.show()