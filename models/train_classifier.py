# import libraries
import sys

import pandas as pd
import numpy as np
import sqlite3
import nltk
import string
import re 
    
from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score
from scipy.stats.mstats import gmean
from sklearn.model_selection import GridSearchCV
import pickle


import warnings
warnings.filterwarnings("ignore")

nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    '''
    Load cleansed data from sqlite database
    Input:
        database_filepath: file path to saved sqlite database
    Output:
        X: messages for categorization
        y: category of message
        category_names: available cateogires
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disasterResponseMD',engine)

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns
  
    return X, Y,category_names


def tokenize(text):
    '''
    Customize tokenization function that normalize, lemmatize and tokenize text
    Input:
        text: input messages
    Output:
        clean_words: normalized, lemmatized, and tokenized text
    '''
    # Convert text to lowercase and remove punctuation
    text = re.sub('[^a-zA-Z0-9]',' ',text)
    
    # tokenize words
    tokens = nltk.word_tokenize(text)
   
    # lemmatize and remove stop words
    stop_words = nltk.corpus.stopwords.words("english")
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return lemmatized



def build_model():
    '''
    Build machine learning pipleines and use GridSearchCV to find the best parameters
    Output:
        cv: model with best parameters
    '''
    
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])


    parameters = {
        'clf__estimator__n_estimators':[50,100],
 #       'clf__estimator__min_samples_split':[2,4],
 #       'clf__estimator__criterion': ['entropy', 'gini']
    }


    cv = GridSearchCV(pipeline,param_grid=parameters,verbose = 2, n_jobs = -1)

    return cv

# def build_model():
#     pipeline = Pipeline([
#         ('vect',CountVectorizer(tokenizer=tokenize)),
#         ('tfidf',TfidfTransformer()),
#         ('clf',MultiOutputClassifier(AdaBoostClassifier()))
#     ])
#     return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
   
    for i, col in enumerate(Y_test):
        print('---------------------------------')
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))
    

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))
   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()