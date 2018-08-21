# NLP_Project

Our project is to classify articles into different predefined categories (b : business , t : science and technology , e : entertainment , m : health) It’s based on supervised learning.

• The dataset consists of Headlines and categories for 400k news items scraped from the web in 2014. 

• This dataset is labeled and contains the following fields:
                    
                    ID Title URL Publisher Category Story Hostname Timestamp 
       
------------------------------------------------------------------------       
              
# Preprocessing

• Drop unwanted columns and keep only title and category fields. 
• Convert titles to lower case. 
• Remove all punctuation. 
• mapping categories into numbers. 
• data encoding. 
• splitting into training and testing. 
• feature extraction

------------------------------------------------------------------------

# Methodology

• NN • KNN • SVM • Decision Tree • Random Forest • SGD classifier

• Naive Bayes 1. Multinomial 2. BernoulliNB 3. GaussianNB • Linear regression • Logistic regression 

------------------------------------------------------------------------

# Experiment 1

2.1 SVM

2.1.1 Definition

Support vector machine is a representation of the training data as points in space separated into categories by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.


• we picked random samples of dataset with about 100 shuffled record • changing maximum number of training document to 20 ones 

                                

# Experiment 2

2.2 KNN

2.2.1 Definition

Neighbors based classification is a type of lazy learning as it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the k nearest neighbors of each point. This algorithm is simple to implement, robust to noisy training data, and effective if training data is large. Need to determine the value of K and the computation cost is high as it needs to computer the distance of each instance to all the training samples.


• we picked random samples of dataset with about 100 shuffled record • changing maximum number of training document to 20 ones gives accuracy of 0.65. • increasing test size increases the accuracy to 0.74 • with k=13 • with k=3 

                                    

# Experiment 3

2.3 NN

2.3.1 Definition

It's same as KNN but the only main different is that k=1


• we picked random samples of dataset with about 100 shuffled record • changing maximum number of training document to 20 ones • increasing test size • with test size=0.1 • with test size =0.33 

                                                 
                                                 
# Experiment 4

2.4 Decision tree

2.4.1 Definition

• Given a data of attributes together with its classes, a decision tree produces a sequence of rules that can be used to classify the data. Decision Tree is simple to understand and visualize, requires little data preparation, and can handle both numerical and categorical data.

• Decision tree can create complex trees that do not generalize well, and decision trees can be unstable because small variations in the data might result in a completely different tree being generated.


• Trying to run with large dataset uci-news-aggregator. • we picked random samples of dataset with about 100 shuffled record • changing maximum number of training document to 20 ones. • Increasing test size 

                                                     

# Experiment 5

2.5 Random forest

2.5.1 Definition

Reduction in over-fitting and random forest classifier is more accurate than decision trees in most cases. Slow real time prediction, difficult to implement, and complex algorithm.


• With test size=0.1 • With test size=0.33 

                                                    

# Experiment 6

2.6 Logistic regression

2.6.1 Definition

Logistic regression is designed for this purpose (classification), and is most useful for understanding the influence of several independent variables on a single outcome variable. Works only when the predicted variable is binary, assumes all predictors are independent of each other, and assumes data is free of missing values.


• we picked random samples of dataset with about 100 shuffled record • changing maximum number of training document to 20 ones • increasing test size • trying to apply with large dataset uci-news-aggregator.

                                                    

# Experiment 7

2.7 Linear regression

2.7.1 Definition

Linear regression is a very simple approach for supervised learning. Though it may seem somewhat dull compared to some of the more modern algorithms, linear regression is still a useful and widely used statistical learning method. Linear regression is used to predict a quantitative response Y from the predictor variable X. Linear Regression is made with an assumption that there’s a linear relationship between X and Y. In linear regression model features should be extracted before splitting as it’s only deals with numerical values


• having a test size=0.33 • reducing size of documents which re used to extract features (max_df=50) • when max_df=2000 

                                                     

# Experiment 8

2.8 gradient descent

2.8.1 Definition Stochastic gradient descent is a simple and very efficient approach to fit linear models. It is particularly useful when the number of samples is very large. It supports different loss functions and penalties for classification.


Advantages: Efficiency and ease of implementation.

Disadvantages: Requires a number of hyper-parameters and it is sensitive to feature scaling.

Experiment Test size=0.33 or 0.2 Exchange hinge loss function with mse 

                                                
                                                
# Experiment 9

2.9 Multi nominal naive bayes

2.9.1 Definition

multinomial is a classification method that generalizes navie bayes to multiclass problems, i.e. with more than two possible discrete outcomes.

test size=0.1 test size=0.33 

                                                        

# Experiment 10

BernoulliNB 2.10

2.10.1 Definition

uses the scikit-learn BernoulliNB estimator (an implementation of the Naive Bayes classification algorithm) to fit a model to predict the value of categorical fields, where explanatory variables are assumed to be binary-valued. This algorithm supports incremental fit.

test size=0.1 

------------------------------------------------------------------------

# Results

1) accuracy=1.0 • accuracy=0.5 • It can't run with large dataset uci-news-aggregator. • it's a big O(n) runtime


2) knn results: • acc=1.0 • acc=0.65 • acc=0.74 • acc=0.62 • acc=0.68 • It can't run with large dataset uci-news-aggregator. • it's a big O(n) runtime


3) nn results: • acc=1 • acc=0.65 • acc=0.75 • acc=0.64 • acc=0.76 • It can't run with large dataset uci-news-aggregator. • it's a big O(n) runtime


4) Decision tree Results:-  Time consuming • Acc=1.0 • Acc=0.44 • Acc=0.98


5) Random forest result: • accuracy =1.0 • accuracy=0.96 • slow real time prediction


6) logistic regression results: • accuracy=1.0 • accuracy=0.44 • accuracy=0.98 • time consuming model


7) accuracy =0.94 • accuracy=0.57 • accuracy =0.92


8) Acc= 0.94 Mse is not supported with gradient descent


9) Accuracy= 0.925 Accuracy= 0.924


10) Accuracy= 0.928 

------------------------------------------------------------------------

# Discussion

1) Decision tree :-
  
                  It has the best accuracy which equals to 0.98 but  it's main problem is time consuming.
                  It also requires a  long depth to get a better accuracy which is too difficult with large data.
                  
                  
2) SVM ,KNN , & NN :-                  
                  
                  they 've the best accuracy which is 100% but the are the most lazy learning models because they've big O of n runtime consumption which means 500K iterations with our dataset.
                  
                  
3) Logistic & linear regression :-                  
                  
                  They've a good accuracy but regression models are used to predict continuous values or amount which is not suitable in our problem.
                  
                  
4) navie bayes :-
                  
                  It gives a good accuracy with any size of dataset . It can learn with a few number of training set.
                  
                  
5) gradient descent :-                  

                  accuracy equals to 0.94 is not bad with a high performance and large dataset.
                  
------------------------------------------------------------------------

# Conclusion

              gradient descent is the best one with our dataset as it's fast and produce higher accuracy.
