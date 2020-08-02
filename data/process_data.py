import sys
import pandas as pd
from IPython.display import display
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #df = pd.concat([messages, categories], axis=1, sort=False)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    
    categories = df['categories'].str.split(";",expand=True)
    row =categories.iloc[0,:]
    categories_colnames= list(map(lambda i: i[ :],row))
    categories_colnames= list(map(lambda i: i[ :-2],categories_colnames))
    #print(categories_colnames)
    categories.columns = categories_colnames
    cat_cp = categories.copy()
    categories = categories.astype(str)
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.strip()[-1])
        categories[column] = pd.to_numeric(categories[column], errors='coerce')
        categories[column] = categories[column].replace(2,1)
    df = df.drop(['categories'],axis =1 )
    df = pd.concat([df, categories], axis=1)
    #boolean = df.duplicated().any()
    df=df.drop_duplicates()
    
    return df
    #boolean1 = df.duplicated().any()
"""    
import imblearn
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0, replacement=True)
X_resampled, y_resampled = rus.fit_resample(X, Y)

print(np.vstack(np.unique([tuple(row) for row in X_resampled], axis=0)).shape)
"""   

def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    #engine = create_engine('sqlite:///../data/{}.db'.format(database_filename))
    
    connection = engine.raw_connection()
    df.to_sql('disaster', connection, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        #disaster_messages.csv, disaster_categories.csv, DisasterResponse.db = sys.argv[1:]
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        #df = load_data(disaster_messages.csv, disaster_categories.csv)
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        #save_data(df, DisasterResponse.db)
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