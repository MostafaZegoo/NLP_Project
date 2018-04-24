import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.metrics import accuracy_score
news_df = pd.read_csv("E:/fcih/second term/NLP/project/uci-news-aggregator/uci-news-aggregator.csv", sep = ",")

news_df['CATEGORY'] = news_df.CATEGORY.map({ 'b': 1, 't': 2, 'e': 3, 'm': 4 })
news_df['TITLE'] = news_df.TITLE.map(
    lambda x: x.lower().translate(str.maketrans('','', string.punctuation))
)
X_train, X_test, y_train, y_test = train_test_split(
    news_df['TITLE'],
    news_df['CATEGORY'],
    random_state =2
)

print("Training dataset: ", X_train.shape[0])
print("Test dataset: ", X_test.shape[0])

#news_df.head()
count_vector = CountVectorizer(stop_words = 'english')
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

naive_bayes = BernoulliNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

print("Accuracy score:", int(accuracy_score(y_test, predictions)*100))