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
import matplotlib.pyplot as plt
# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
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
    cod = nb.predict(vectorizer.transform([title]))
    return cat_names[encoder.inverse_transform(cod)[0]]
  
''' grab the data
#About this Dataset
#
#This dataset contains headlines, URLs, and categories for 422,937 news stories collected by a web aggregator between March 10th, 2014 and August 10th, 2014.
#
#News categories included in this dataset include business; science and technology; entertainment; and health. Different news articles that refer to the same news item (e.g., several articles about recently released employment statistics) are also categorized together.
#Content
#
#The columns included in this dataset are:
#
#    ID : the numeric ID of the article
#    TITLE : the headline of the article
#    URL : the URL of the article
#    PUBLISHER : the publisher of the article
#    CATEGORY : the category of the news item; one of: -- b : business -- t : science and technology -- e : entertainment -- m : health
#    STORY : alphanumeric ID of the news story that the article discusses
#    HOSTNAME : hostname where the article was posted
#    TIMESTAMP : approximate timestamp of the article's publication, given in Unix time (seconds since midnight on Jan 1, 1970) '''

news = pd.read_csv("data/uci-news-aggregator.csv")
# let's take a look at our data
print news.head()
#Normalize the title
news['TEXT'] = [normalize_text(s) for s in news['TITLE']]
print news.head()
news['CATEGORY'].unique()
print news['CATEGORY'].value_counts()
'''Now we know that the categories are not evenly distributed:
e    152469
b    115967
t    108344
m     45639
Which means we have bearely 45000 news articles under health category. For most statistical calculations, it wouldn't matter but for neural nets it will.    '''
news['CATEGORY'].value_counts().plot(kind="bar")
plt.show()


# pull the data into vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(news['TEXT'])

encoder = LabelEncoder()
y = encoder.fit_transform(news['CATEGORY'])

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# take a look at the shape of each of these
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#So the x training vector contains 337935 observations of 54637 occurrences -- this latter number is the number of unique words in the entire collection of headlines. The x training vector contains the 337935 labels associated with each observation in the x training vector.

nb = MultinomialNB()
nb.fit(x_train, y_train)
results_nb_cv = cross_val_score(nb, x_train, y_train, cv=10)
print(results_nb_cv.mean())


nb.score(x_test, y_test)
x_test_pred = nb.predict(x_test)
confusion_matrix(y_test, x_test_pred)
print(classification_report(y_test, x_test_pred, target_names=encoder.classes_))
print predict_cat("Titanic: a good movie")