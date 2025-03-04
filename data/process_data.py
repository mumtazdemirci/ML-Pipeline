import sys

# import libraries

import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and its categories labels
    Input:
        messages_filepath: path to messages.csv file
        categories_filepath: path to categories.csv file
    Output:
        df: merged dataframe contains messages and its category
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages,categories,left_on = 'id', right_on = 'id')

    return df


def clean_data(df):
    '''
    Clean data to prepare for future ML task
    Input:
        df: merged dataframe from load_data() function
    Output:
        df: cleansed dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    
    category_colnames = row.apply(lambda x:x.split('-')[0])

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    categories = categories.applymap(lambda x: int(x[-1]))


    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1,join='inner')

    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Save data to sqlite database
    Input:
        df: cleansed dataframe
        database_filename: file name for the db
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disasterResponseMD', engine, index=False)    

    
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