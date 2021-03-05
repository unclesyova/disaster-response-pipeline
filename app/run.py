import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
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

# load data
engine = create_engine('sqlite:///../data/Classification.db')
df = pd.read_sql_table('message_category', engine)
prec_df = pd.read_sql_table('weighted_precision', engine)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts = df.iloc[:,5:].sum()
    category_names = list(category_counts.index)

    categories = prec_df['index'].values
    precisions = list(prec_df['0'].values)
 
    # create visuals
    val = df.groupby('genre').count()
    key = [x for x in val.index]

    # create visuals
    graphs = [dict(
        data=[Bar(
            x=category_names,
            y=category_counts,
        )],
        layout=dict(
            title='Dataset Category Distribution',
            yaxis=dict(
                title="Category Counts",
            ),
            xaxis=dict(
                title="Category",
                tickangle=30
            )
        )
    ), dict(
        data=[Bar(
            x=genre_names,
            y=genre_counts
        )],
        layout=dict(
            title='Bar Plot of Messages By Genre',
            yaxis=dict(
                title="Messages"
            ),
            xaxis=dict(
                title="Genres"
            )
        )
    ), dict(
        data=[Bar(
            x=categories,
            y=precisions
        )],
        layout=dict(
            title='Precision of Prediction by Category',
            yaxis=dict(
                title="Precision"
            ),
            xaxis=dict(
                tickangle=30,
                title="Category"
            )
        )
    )

   ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
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
