# import libraries
import numpy as np
import pandas as pd
import sys
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load data from specified sources and combine it into one dataframe

    Keyword arguments:
    messages_filepath -- the filepath to messages data
    categories_filepath -- the filepath to categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner', on='id')

    return df


def clean_data(df):
    """Clean the dataframe obtained after load_data() function

    Keyword arguments:
    df -- the dataframe obtained by merging messages and categories data
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True) 

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda name : name[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    # drop invalid values in `related` column
    df = df[df['related'] != 2]

    return df



def save_data(df, database_filename):
    """Save the dataframe as a sql table into specified database file

    Keyword arguments:
    df -- a dataframe
    database_filename -- a database file to save the dataframe into
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('message_category', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
