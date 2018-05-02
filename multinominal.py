import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import string
import matplotlib.pyplot as plt

news_df = pd.read_csv("uci-news-aggregator.csv", sep = ",")

news_df['CATEGORY'] = news_df.CATEGORY.map({ 'b': 1, 't': 2, 'e': 3, 'm': 4 })
news_df['TITLE'] = news_df.TITLE.map(lambda x: x.lower().translate(str.maketrans('','', string.punctuation)))
X_train, X_test, y_train, y_test = train_test_split(news_df['TITLE'],news_df['CATEGORY'],test_size=0.1,random_state =42)

"""count_vector = CountVectorizer(stop_words = 'english')
training_data = count_vector.fit_transform(X_train.values)
testing_data = count_vector.transform(X_test.values)"""

tf_vector = TfidfVectorizer(stop_words = 'english')
training_data = tf_vector.fit_transform(X_train.values)
testing_data = tf_vector.transform(X_test.values)

naive_bayes = MultinomialNB()
hist=naive_bayes.fit(training_data, y_train)

y_pred = naive_bayes.predict(testing_data)


tn,fn,tp,fp=confusion_matrix(y_test,y_pred)

print(confusion_matrix(y_test,y_pred))
print("===================================")
print(classification_report(y_test,y_pred))
print("===================================")
print("Accuracy score:",accuracy_score(y_test,y_pred))
print("===================================")
print(naive_bayes.predict(tf_vector.transform(["nescafe is a product from nestle"])))

plt.plot(tn)
plt.plot(tp)
plt.plot(fn)
plt.plot(fp)
plt.show()