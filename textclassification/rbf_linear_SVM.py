# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 17:37:56 2018

@author: sshekhar
"""

# get some libraries that will be useful
import re
import string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
from sklearn.model_selection import GridSearchCV
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
# function for encoding categories
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

#Function to normalize the text
def normalize_text(s):
    #lower-case the text
    s = s.lower()
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    for ch in string.punctuation:
        s = s.replace(ch, " ")
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    s = re.sub("[0-9]+", "||DIG||",s)
    s = re.sub(' +',' ', s)
    return s

#Function to predict the category for a given title
def predict_cat(title):
    title=title.lower()
    cat_names = {'b' : 'business', 't' : 'science and technology', 'e' : 'entertainment', 'm' : 'health'}
    clf_pred = clf.predict(vectorizer.transform([title]))
    return cat_names[encoder.inverse_transform(clf_pred)[0]]
  

news = pd.read_csv("data/uci-news-aggregator.csv")
# let's take a look at our data
#Normalize the title
news['TEXT'] = [normalize_text(s) for s in news['TITLE']]
news['CATEGORY'].unique()

# pull the data into vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(news['TEXT'])
encoder = LabelEncoder()
y = encoder.fit_transform(news['CATEGORY'])

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# Instantiate the classifier: clf
#Since the dataset is small enough, non-linear SVM should perform equally well
#clf = svm.SVC(decision_function_shape='ovo', verbose=1)
clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5, n_jobs=-1)

#lin_clf = svm.LinearSVC()
# Fit the classifier to the training data
clf.fit(x_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_params_)
print("Grid scores on training set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

#lin_clf.fit(x_train, y_train)
# Print the accuracy
print("Accuracy: {}".format(clf.score(x_test, y_test)))
#print("Linear SVM Accuracy: {}".format(lin_clf.score(x_test, y_test)))
x_test_clv_pred = clf.predict(x_test)
confusion_matrix(y_test, x_test_clv_pred)
print(classification_report(y_test, x_test_clv_pred, target_names=encoder.classes_))

'''
 Apart from accuracy, three major metrics to understand the task for classification are: precision, recall and f1-score.

Precision: The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

Recall: The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

F1-Score: It can be interpreted as a weighted harmonic mean of the precision and recall, where an f1-score reaches its best value at 1 and worst score at 0.

Support: Although not a scoring metric, it is an important quantity when looking at different metrics. It is the number of occurrences of each class in y_true.
'''

randomtitle="vehicular pollution - a big hazard for children"
print predict_cat(randomtitle)
