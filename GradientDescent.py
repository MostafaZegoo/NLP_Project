import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import string

np.random.seed(123456)
news_df = pd.read_csv("uci-news-aggregator.csv", sep = ",")
news_df['CATEGORY'] = news_df.CATEGORY.map({ 'b': 1, 't': 2, 'e': 3, 'm': 4 })
news_df['TITLE'] = news_df.TITLE.map(lambda x: x.lower().translate(str.maketrans('','', string.punctuation)))

X_train, X_test, y_train, y_test = train_test_split(news_df['TITLE'], news_df['CATEGORY'], test_size=0.3,random_state=42)


count_vector = CountVectorizer(stop_words = 'english')
training_data = count_vector.fit_transform(X_train.values)
testing_data = count_vector.transform(X_test.values)

encoder=LabelEncoder()
encoder.fit(y_train)
encoder.transform(y_train)

sgd=SGDClassifier(loss="hinge", shuffle=True, random_state=42)
sgd.fit(training_data,y_train)
y_pred=sgd.predict(testing_data)

tn,fn,tp,fp=confusion_matrix(y_test,y_pred)

print(confusion_matrix(y_test,y_pred))
print("===================================")
print(classification_report(y_test,y_pred))
print("===================================")
print("Accuracy score:",accuracy_score(y_test,y_pred))
print("===================================")
print(sgd.predict(count_vector.transform(["nescafe is a product from nestle"])))

plt.plot(tn)
plt.plot(tp)
plt.plot(fn)
plt.plot(fp)
plt.show()