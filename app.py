import pickle
from flask import Flask, send_file
from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
import string
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import string
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify

import string



app = Flask(__name__)

with open('updated_sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)






@app.route('/')
def index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            print("Here")
            sentence = request.form['sentence']

            # Preprocess the input sentence
            processed_sentence = preprocess(sentence)
            # Tokenize the processed sentence

            tokenized_sentence = tokenize(processed_sentence)
            # Use the model to predict sentiment
            print(tokenized_sentence)
            prediction = model.predict([tokenized_sentence])
            print("here--------------->",prediction)
            # Return the prediction
            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e)})


def preprocess(sentence):
    # Convert to lowercase
    sentence = sentence.lower()
    # Remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespaces
    sentence = ' '.join(sentence.split())
    print(sentence)
    # Additional preprocessing steps can be added here
    return sentence

def tokenize(sentence):
    # Tokenize the sentence using NLTK's word_tokenize function
    tokens = word_tokenize(sentence)
    return tokens


if __name__ == '__main__':
    app.run(debug=True)
