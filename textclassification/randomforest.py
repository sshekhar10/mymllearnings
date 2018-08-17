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
from sklearn.ensemble import RandomForestClassifier
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
# Instantiate the classifier: clf
clf = RandomForestClassifier()
# Fit the classifier to the training data
clf.fit(x_train, y_train)
# Print the accuracy
print("Accuracy: {}".format(clf.score(x_test, y_test)))
x_test_clv_pred = clf.predict(x_test)
confusion_matrix(y_test, x_test_clv_pred)
print(classification_report(y_test, x_test_clv_pred, target_names=encoder.classes_))

randomtitle="vehicular pollution - a big hazard for children"
print predict_cat(randomtitle)
