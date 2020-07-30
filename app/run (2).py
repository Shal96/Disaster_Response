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
import plotly.graph_objs as go
from plotly.graph_objs import Bar,Pie
import plotly.express as px
from sklearn.externals import joblib
from sqlalchemy import create_engine


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

@app.route('/')
@app.route('/index')
@app.route('/index.html')
@app.route('/#')

def index():
   
    c=df.drop(df.columns[:4],axis=1)
    count = c.sum().sort_values(ascending =False)
    #print(count)

    cols = count.head()
    df1 = pd.DataFrame({'categories':cols.index, 'values':cols.values})
    df1.loc[df1['values'] < 1000, 'categories'] = 'Other categories'
    pie_chart=px.pie(df1, names='categories', values='values')
    
    
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending = False)
    print(genre_counts)
    
    genre_counts = pd.DataFrame({'genre':genre_counts.index, 'count':genre_counts.values})
    bar_chart=[]
    for genre in genre_counts['genre']:
        x_values = list(genre_counts[genre_counts['genre'] == genre]['genre'])
        y_values = list(genre_counts[genre_counts['genre'] == genre]['count'])
        bar_chart.append(Bar(x=x_values, y=y_values, showlegend = True))
    
    graphs = [{'data': pie_chart,
              'layout':[{'title':'Pie_Chart'}]},
        
        {'data':bar_chart, 
           
           'layout':{'title':'Distribution',
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