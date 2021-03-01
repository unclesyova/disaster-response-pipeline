# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

# Import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
from joblib import dump
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import sys


def load_data(database_filepath):
    """Load data from sql table as a dataframe.
    Return messages, categories as vectors and category names.

    Keyword arguments:
    database_filepath -- filepath to the database from which to take data
    """
    # convert sql table into a dataframe
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message_category', engine)

    X = df['message'].values
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names



def tokenize(text):
    """Replace url in a message with `url_placeholder` 
    tokenize and lemmatize message
    normalize case and delete leading/trailing whitespaces

    Keyword arguments:
    text -- a message to tokenize
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok.strip().lower(), pos='v')
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    """Build a pipeline for ML model and set parameters for grid search.
    Return the model.
    """
    # Build a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier())),
    ]) 

    # Perform grid search
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv 


def evaluate_model(model, X_test, Y_test, category_names):
    """Make prediction on X_test and 
    evaluate every predicted field with classification report
    
    Keyword arguments:
    model -- ML model for prediction
    X_test -- test set of messages
    Y_test -- categories corresponiding to X_test
    category_names -- the names of categories
    """
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns, index=Y_test.index)

    for col1, col2 in zip(Y_test.columns, Y_pred.columns):
        print(col1+':')
        print(classification_report(Y_test[col1], Y_pred[col2]))


def save_model(model, model_filepath):
    """Save the obtained model as a joblib file

    Keyword arguments:
    model -- ML model
    model_filepath -- the filepath to save the model
    """
    dump(model, model_filepath)




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
