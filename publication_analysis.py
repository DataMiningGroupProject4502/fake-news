'''
Code to explore the question #5 "Are certain publications more prone to fake news than others?"

'''

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px
import nltk
import re
import sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Load Data
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")
harvard = pd.read_csv("articles.csv")

# Data Preprocessing Steps
# Fix categorical data  
true = true.iloc[:, [0, 1]]
fake = fake.iloc[:, [0, 1]]
true['label'] = '0'    
fake['label'] = '1'    


true = true.iloc[:, [0, 1, 2]]
fake = fake.iloc[:, [0, 1, 2]]
harvard = harvard.iloc[:, [1, 2]]


# 1. Data Integration
train_dataset = pd.concat([true, fake]).reset_index(drop = True)                  # Dataset to train model
master_dataset = harvard
'''
print(train_dataset)
print("------------------------------------")
print(master_dataset)
'''

# Combine title and text
train_dataset['original'] = train_dataset['title'] + ' ' + train_dataset['text']
#master_dataset['original'] = master_dataset['name'] + ' ' + master_dataset['content']
master_dataset['original'] = master_dataset['name']

print("True Fake Training Dataset")
print(train_dataset)
print("------------------------------------")
print("Master Dataset")
print(master_dataset)

x_train = train_dataset.loc[:, 'original']
y_train = train_dataset.loc[:, 'label']

#CountVectorizer - transforms text into a vector on the basis of frequency
#TfidTransformer - Transforms vector to a normalized tf
#MultinomialNB - Multinomial Naive bayes classifier
pipe = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('model', MultinomialNB())])


model = pipe.fit(x_train, y_train)


x_test = master_dataset['original'].values.astype(str)
predictions = model.predict(x_test)

submission = pd.DataFrame({
    'Id' : master_dataset.index,
    'Source': master_dataset.source,
    'Predicted': predictions
})

frequency = submission.groupby(['Source', 'Predicted']).size()

print(submission)
print(frequency)



submission.to_csv('harvard_pub_predicitions.csv', index=False)