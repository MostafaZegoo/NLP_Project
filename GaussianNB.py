import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import string
import itertools
news_df = pd.read_csv("uci-news-aggregator.csv", sep = ",")

news_df['CATEGORY'] = news_df.CATEGORY.map({ 'b': 1, 't': 2, 'e': 3, 'm': 4 })
news_df['TITLE'] = news_df.TITLE.map(lambda x: x.lower().translate(str.maketrans('','', string.punctuation)))
X_train, X_test, y_train, y_test = train_test_split(news_df['TITLE'],news_df['CATEGORY'],test_size=0.1,random_state =42)

"""count_vector = CountVectorizer(stop_words = 'english')
training_data = count_vector.fit_transform(X_train.values)
testing_data = count_vector.transform(X_test.values)"""

tf_vector = TfidfVectorizer(stop_words = 'english',max_df=10)
training_data =tf_vector.fit_transform(X_train)
testing_data = tf_vector.transform(X_test)

naive_bayes = GaussianNB()
naive_bayes.fit(training_data, y_train)

y_pred = naive_bayes.predict(testing_data)
print(y_pred)
print("Accuracy score:", int(accuracy_score(y_test,y_pred)*100))
print(naive_bayes.predict(tf_vector.transform(["nescafe is a product from nestle"])))
tp,tn,fp,fn=confusion_matrix(y_test,y_pred)

print("Accuracy score:",accuracy_score(y_test,y_pred))
print("===================================")
print(classification_report(y_test,y_pred))
print("===================================")

numerical_result=naive_bayes.predict(tf_vector.transform(["nescafe is a product from nestle"]))
result=""
if numerical_result == 1:
    result += "business"
elif numerical_result == 2:
    result += "science and technology"
elif numerical_result == 3:
    result += "entertainment"
elif numerical_result == 4:
    result += "health"
print(result)
plt.plot(tp)
plt.plot(tn)
plt.plot(fp)
plt.plot(fn)
plt.legend(['true positives', 'true negatives','false positives','false negatives'], loc='upper left')
plt.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
plot_confusion_matrix(cm = np.array([tp,tn,fp,fn]),
                      normalize = False,
                      target_names = ['business', 'entertainment', 'healt','science and technology'],
                      title        = "Confusion Matrix")