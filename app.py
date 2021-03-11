from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from keras.models import load_model
import data_preprocessing as dp
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
stopwords=set(stopwords.words('english'))

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def init():
    global model,graph
    model = load_model('sentiment_analysis.h5')
    graph = tf.compat.v1.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("home.html")

@app.route('/sentiment_analysis_prediction', methods = ['POST', "GET"])
def sent_anly_prediction():
    if request.method=='POST':
        text = request.form['text']
        max_review_length = 500
        def data_cleaning(x):
            x = str(x).lower().replace('/', '').replace('_', ' ').replace('<br />', ' ')
            x = dp.cont_exp(x)  # expansion of sentences
            x = dp.remove_urls(x)
            x = dp.remove_html_tags(x)
            x = dp.remove_accented_chars(x)
            x = dp.remove_special_chars(x)
            x = re.sub("(.)\1{2,}", "\1", x)  # removal of multi charcters in data
            return x
        text = data_cleaning(text)
        def encod(z):
            z = word_tokenize(z)
            new_words = []
            for w in z:
                if w not in stopwords:
                    new_words.append(w)
            my_vocab_size = 100000
            enc = [one_hot(i, my_vocab_size) for i in new_words]
            return enc
        x_t = encod(text)
        x_t = sequence.pad_sequences(x_t, maxlen=max_review_length)
        cls = model.predict_classes(x_t)
        prob=model.predict(x_t)
        with graph.as_default():
            probability = prob[4][0]
            class1 = cls[4][0]
        if class1 == 0:
            sentiment = 'Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png')
        else:
            sentiment = 'Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')
    return render_template('home.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)

if __name__ == "__main__":
    init()
    app.run()