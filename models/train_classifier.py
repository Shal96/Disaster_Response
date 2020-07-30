import sys

#import sqlalchemy as db
from sqlalchemy import create_engine

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_classification

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from scipy.stats import gmean
import pickle


def load_data(database_filepath):
    
    engine = create_engine('sqlite:///{}.db'.format(database_filepath), pool_recycle=3600)
    connection = engine.connect()
    df = pd.read_sql_table('disaster',connection)
    
    X = df.message.values
    
    Y = df.loc[:, df.columns]
    Y= df.drop(['id', 'message', 'original', 'genre'], axis =1)
    Y=Y.astype(int)
    
    return X,Y
    
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
    


def build_model():
    
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf',TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    return pipeline


def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    #confusion_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), labels=labels)
    classification_rep = classification_report(y_test, y_pred)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    #print("Confusion Matrix:\n", confusion_mat)
    print("Classifcation Report:\n", classification_rep)
    print("Accuracy:", accuracy)

    
def train_evaluate_model(model, X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model_fit = model.fit(X_train, y_train)
    
    """
    parameters = {'clf__estimator__n_estimators': [50,100, 200],
                  'clf__estimator__min_samples_split':[2, 3, 4],
                  'vect__max_df': (0.5, 0.75, 1.0),
                  'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.5, 0.75, 1.0),
                  'vect__max_features': (None, 5000, 10000)}

    cv = GridSearchCV(model, param_grid=parameters,verbose = 2, n_jobs = -1)
    cv.fit(X_train, y_train)
    print("\nBest Parameters:", cv.best_params_)
    """
    y_pred = model_fit.predict(X_test)
    
    get_results = display_results(y_test, y_pred)
    print(get_results)
    return X_train, X_test, y_train, y_test, y_pred, model, get_results
       
def save_model(model, model_filepath):
    #with open('../models/{}.pickle'.format(model_filepath), 'wb') as f:
        #pickle.dump(model, f)
        
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
   
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, Y = load_data(database_filepath)
        
        
        print('Building model...')
        model = build_model()
        grid=model.get_params().keys()
        print(grid)
        
        #print('Training model...')
        #model.fit(X_train, Y_train)
        
        print('Training and Evaluating model...')
        train_evaluate_model(model, X, Y)
        
        #print('Display Results....')
        #display_results()

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
