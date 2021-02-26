from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle
import en_core_web_md
nlp = en_core_web_md.load() 

# load the model from disk
app = Flask(__name__)
filename = 'model11.pkl'
clf = pickle.load(open(filename, 'rb'))
# cv = pickle.load(open('transform.pkl', 'rb'))
# tfidf_vectorizer = TfidfVectorizer()

def get_vec(x):
  doc = nlp(x)
  return doc.vector

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        data=[message]
        comment = pd.DataFrame([message])
        comment = comment[0].apply(lambda x: get_vec(x))
        XX = comment.to_numpy()
        XX = XX.reshape(-1,1)
        XX = np.concatenate(np.concatenate(XX,axis = 0),axis = 0).reshape(-1,300)
        data=XX
        # tfidf = tfidf_vectorizer.fit_transform(data)
        # vect = cv.transform(data).toarray()
        my_prediction = clf.predict(data)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)