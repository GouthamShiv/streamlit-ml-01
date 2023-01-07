#
# Created on: Fri Jan 06 2023 16:50:17
# Created by: Goutham, Shivanna
#


import streamlit as sl
import numpy as np
from sklearn import datasets as ds
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CLASSIFIER_KNN = 'KNN'
CLASSIFIER_LR = 'Logistic Regression'
CLASSIFIER_DTC = 'Decision Tree Classifier'
CLASSIFIER_NB = 'Naïve Bayes - GaussianNB'
CLASSIFIER_RFC = 'Random Forest Classifier'

sl.set_page_config(page_title='ML classifiers comparison', layout='wide')

sl.header('''Streamlit app to visually compare ML classifiers from SKLearn.
---''')
sl.write('''
###### Below are the list of datasets and classifiers available, these are not 1:1 mapping, so, they can be mixed and matched from the selection dropdowns in the sidebar.
|Datasets|Classifiers|
|---|---|
|Iris|Naïve Bayes - GaussianNB|
|Wine|KNN Classifier|
|Breast Cancer|Logistic Regression|
||Decision Tree Classifier|
||Random Forest Classifier|

---
''')

sl.sidebar.write('''### Parameters''')

available_datasets = {'Iris': ds.load_iris(), 'Wine': ds.load_wine(
), 'Breast Cancer': ds.load_breast_cancer()}
available_classifiers = [CLASSIFIER_KNN, CLASSIFIER_LR,
                         CLASSIFIER_DTC, CLASSIFIER_NB, CLASSIFIER_RFC]

chosen_dataset = sl.sidebar.selectbox(
    label='Choose a dataset', options=available_datasets.keys())

chosen_classifier = sl.sidebar.selectbox(
    label='Choose a classifier', options=available_classifiers)

sl.write('### You have chosen `', chosen_dataset,
         '` dataset and `', chosen_classifier, '` as the classifier.')

for available_dataset in available_datasets.keys():
    if chosen_dataset == available_dataset:
        loaded_dataset = available_datasets.get(chosen_dataset)

if chosen_classifier == CLASSIFIER_KNN:
    k = sl.sidebar.slider(label='Choose \'k\' value',
                          min_value=1, max_value=15)
    classifier = KNeighborsClassifier(n_neighbors=k)
elif chosen_classifier == CLASSIFIER_LR:
    max_iter = sl.sidebar.slider(
        label='Choose \'max_iter\' value', min_value=5, max_value=100, step=5)
    classifier = LogisticRegression(max_iter=max_iter)
elif chosen_classifier == CLASSIFIER_DTC:
    criterion = sl.sidebar.selectbox(
        label='Choose a \'criterion\'', options=['gini', 'entropy'])
    classifier = DecisionTreeClassifier()
elif chosen_classifier == CLASSIFIER_NB:
    classifier = GaussianNB()
elif chosen_classifier == CLASSIFIER_RFC:
    max_depth = sl.sidebar.slider(
        label='Choose \'max_depth\' value', min_value=2, max_value=15)
    n_estimators = sl.sidebar.slider(
        label='Choose \'n_estimators\' value', min_value=1, max_value=100)
    classifier = RandomForestClassifier(
        max_depth=max_depth, n_estimators=n_estimators, random_state=1992)

X = loaded_dataset.data
y = loaded_dataset.target
sl.write('Shape of the dataset:', X.shape)
sl.write('Number of classes in dataset:', len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1992)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

sl.write('Classifier `', chosen_classifier,
         '` in current hyperparameter setting is ', accuracy, ' accurate')
