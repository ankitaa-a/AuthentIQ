# This is a sample Python script.
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from newsapi import NewsApiClient
import flask,requests
from sklearn.linear_model import PassiveAggressiveClassifier


vector = TfidfVectorizer(stop_words="english", max_df=0.7)

app = flask.Flask(__name__)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def train_and_save_model():
    news_articles = pd.read_csv(r'C:\Users\Admin\Downloads\news.csv')

    # Combining datasets
    labels = news_articles['label']

    # Splitting data into training and testing
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(news_articles['text'], labels, test_size=0.2, random_state=20)

    vector = TfidfVectorizer(stop_words="english", max_df=0.7)
    tf_train = vector.fit_transform(Xtrain)
    tf_test = vector.transform(Xtest)

    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tf_train, Ytrain)

    y_pred = pac.predict(tf_test)

    acc = accuracy_score(Ytest, y_pred)

    print(f'Accuracy: {acc}')
    print(confusion_matrix(Ytest, y_pred, labels=['FAKE', 'REAL']))

    filename = "finalmodel.pkl"
    pickle.dump(pac, open(filename, 'wb'))
    return pac, vector



model,vector=train_and_save_model()

@app.route("/")
def home():
    return flask.render_template("index.html")

@app.route("/prediction", methods=['GET','POST'])
def prediction():
    if flask.request.method == "POST":
        news = flask.request.form['hero-field']
        predict = model.predict(vector.transform([news]))
        result = "Fake" if predict[0] == 'FAKE' else "Real"  # Assuming 'FAKE' corresponds to fake news
        return result

if __name__ == '__main__':
    print_hi('PyCharm')

    app.run()