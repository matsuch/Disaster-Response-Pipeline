# import libraries
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC

from sqlalchemy import create_engine

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    
    """
    Function:
    load data from database
    Args:
    database_filepath: the path of the database
    Return:
    X (DataFrame) : Message features dataframe
    y (DataFrame) : target dataframe
    category (list of str) : target labels list
    """
    
    engine = create_engine('sqlite:///db_disaster_messages.db')
    df = pd.read_sql_table('disaster_massages_table', engine) 

    X = df['message'] #message column
    y = df.iloc[:, 4:] #classification label
    return X, y


def tokenize(text):
    
    """
    Function: split text into words and return the root form of the words
    Args:
      text(str): the message
    Return:
      lemm(list of str): a list of the root form of the message words
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens

def build_model():
    
    """
     Function: build a model for classifing the disaster messages
     Return:
       cv(list of str): classification model
    """

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))    
    ])
    
    parameters = {
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__n_estimators': [50, 100, 200]

    }   
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=4, n_jobs=-1)
    
    return cv    

def evaluate_model(model, X_test, y_test):
    """
    Function: Evaluate the model and print the f1 score, precision and recall for each output category of the dataset.
    Args:
    model: the classification model
    X_test: test messages
    y_test: test target
    """
            
    y_pred = model.predict(X_test)
    overall_accuracy = (y_pred == y_test).mean().mean()*100
    
    print('Model overall Accuracy: ', overall_accuracy, '\n')
    y_pred_df = pd.DataFrame(y_pred, columns = y_test.columns)
    for column in y_test.columns:
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(y_test[column], y_pred_df[column]))

def save_model(model, model_filepath):   
    """
    Function: Save a pickle file of the model
    Args:
    model: the classification model
    model_filepath (str): the path of pickle file
    """
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

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
