import json
import plotly
import plotly.colors
import pandas as pd
import joblib
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

#sys.path.insert(0,'../data')
#sys.path.insert(0,'../models')
#print(sys.path)


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model

#@app.route('/')
#@app.route('/index')


def count_col():
    c = df.drop(df.columns[:4], axis=1)
    print(c.columns)

    count = c.sum().sort_values(ascending = False)
    
    return count

@app.route('/')
@app.route('/index')
@app.route('/index.html')
@app.route('/#')

def index():
    
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending = False)
    genre_counts = pd.DataFrame({'genre':genre_counts.index, 'count':genre_counts.values})
    
    for genre in genre_counts['genre']:
        x_val = genre_counts[genre_counts['genre'] == genre]['genre'].tolist()
        y_val = genre_counts[genre_counts['genre'] == genre]['count'].tolist()
    
    
    pop_cols = count_col().head()
    pop_cols_df = pd.DataFrame({'categories':pop_cols.index, 'values':pop_cols.values})
    print(pop_cols_df)
    
    for i in pop_cols_df['categories']:
        x_values = list(pop_cols_df[pop_cols_df['categories'] == i]['categories'])
        y_values = list(pop_cols_df[pop_cols_df['categories'] == i]['values'])
        print(x_values, y_values)
        
    graphs = [{'data':[Bar(x=x_values,
                    y=y_values,
                    marker = dict(color = i),
                    name = i,
                    showlegend = True)], 
           
           'layout':{'title':'Distribution',
                     'yaxis':{'title':'Number of Messages'},
                     'xaxis':{'title':'Category'}}},
              
              {'data':[Bar(x=x_val,
                    y=y_val,
                    marker = dict(color = genre),
                    name = genre,
                    showlegend = True)], 
           
           'layout':{'title':"Message Genre Counts in Dataset",
                     'yaxis':{'title':'Number of Messages'},
                     'xaxis':{'title':'Genre'}}}]
    
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
#@app.route('/go')

@app.route('/go')
@app.route('/go.html')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()