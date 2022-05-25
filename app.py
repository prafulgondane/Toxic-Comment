from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import re
import string
# https://www.tutorialspoint.com/flask
from nltk.stem import WordNetLemmatizer
import flask
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from zipfile import ZipFile

import nltk
nltk.download('wordnet')

app = Flask(__name__)

print(tf.__version__)

#pip install tensorflow

#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

# loading
with open('stopwords.pickle', 'rb') as handle:
    stopwords = pickle.load(handle)
    print(len(stopwords))
    
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    print('tokenizer loaded')
    
with ZipFile('bi_lstm_model.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()  
# loading model
model = keras.models.load_model('bi_lstm_model.h5')  
    
# load json and create model
#from keras.models import model_from_json
#json_file = open('bi-lstm-model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
# load weights into new model
#model.load_weights("bi-lstm-model.h5")
#print("Loaded model from disk")
    
print('model loaded')


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/get_loan_details', methods=['POST'])
def get_loan_details():
    print('Inside get_loan_details')
    to_predict_list = request.form.to_dict()
    id = to_predict_list['loan_id']
    d = data.loc[data['SK_ID_CURR'] == int(id)]
    print('Data :', d)
    gender = "M" if int(d['CODE_GENDER_L'].values[0]) > 0 else "F"
    loan_id = str(d['SK_ID_CURR'].values[0])
    income = str(d['AMT_INCOME_TOTAL'].values[0])
    loan_amt = str(d['AMT_CREDIT'].values[0])
    print('Data :', loan_id, gender,income , loan_amt)
    #return jsonify({'LOAN_ID' : d['SK_ID_CURR'].values[0], 'GENDER': gender , 'INCOME': d['AMT_INCOME_TOTAL'].values[0], 'LOAN_AMT' : d['AMT_CREDIT'].values[0]})
    return jsonify({'loan_id': loan_id, 'gender' : gender, 'income' : income, 'loan_amt' : loan_amt})

@app.route('/predict', methods=['POST'])
def predict():
    print('Inside predict')
    
    print(request)
    print(request.form.to_dict())
    to_predict_list = request.form.to_dict()
    
    
    id = to_predict_list['comment']
    print('comment :', id)
    data = text_preprocessing(id)
    print('comment :', data)
    # All comments must be truncated or padded to be the same length.
    MAX_SEQUENCE_LENGTH = 250
    
    data_list = []
    flat_tokens = tokenizer.texts_to_sequences(data)
    flat_tokens = [item for sublist in flat_tokens for item in sublist]
    data_list.append(flat_tokens)
    print('Data :', data_list)
    # Prepare data
    train_data = pad_sequences(data_list, maxlen=MAX_SEQUENCE_LENGTH)
    
    print('Data :', train_data)
 
    prediction = model.predict(train_data)[:, 1]
    isToxic = ['TOXIC' if prediction[0] > 0.5 else 'NON TOXIC']
    prediction[0] = float(f'{prediction[0]:.2f}')
    print("prediction : ",prediction , isToxic)
    return jsonify({'prediction': prediction.tolist(), 'isToxic':isToxic})


#defining function for tokenization
def tokenization(text):
    tokens = text.split()
    return tokens

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords[0:140]]  # after 140 indext it will don't haven't kind of words
    return output

#defining the function for lemmatization
def lemmatizer(text):
  lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
  return lemm_text

def text_preprocessing(x):
  #Remove punctuations
  x = remove_punctuation(x)
  print("remove_punctuation : ", x)
  #make lower case
  x=x.lower()
  print("lower : ", x)
  #toenize the string
  x=tokenization(x)
  print("tokenization : ", x)
  #remove stop words
  x=remove_stopwords(x)
  print("remove_stopwords : ", x)
  #lemmatize
  x=lemmatizer(x)
  print("lemmatizer : ", x)
  return x

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9082)
