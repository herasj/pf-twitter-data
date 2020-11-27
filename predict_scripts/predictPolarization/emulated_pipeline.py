#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:08:34 2020

@author: kodewill
"""
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import re
from nltk.corpus import stopwords # Usado para eliminar las stopwords
import stanza 
stanLemma = stanza.Pipeline(processors='tokenize,mwt,pos,lemma', lang='es', use_gpu=False)
from nltk.tokenize import word_tokenize # Word to tokens
import keras.backend as K

threshold = 0.5

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def load_obj(name):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


#Get rid of punctuation 
def removePunctuation(tweet: str)->str:
    tweet = re.sub(r'[^\w\s]', '', tweet)
    return tweet

#Remove links
def removeLinks(tweet:str)-> str:
    tweet = re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', '',tweet)
    return tweet

#Remove hashtag and mentions
def removeHashtag(tweet:str)-> str:
    tweet = re.sub('(#)+(\w|\d)+', '',tweet)
    tweet = re.sub('(@)+(\w|\d)+', '',tweet)
    return tweet

#Remove accent marks
def removeAccentMarks(texto: str)->str:
    finalText = ""
    for word in texto:
        finalText += word.upper().replace('Á','A').replace('É','E').replace('Í','I').replace('Ó','O').replace('Ú','U').replace('Ü','U')
    return finalText

# Remove spanish's stopwords.
def removeStopWords(sentence: str):
    # Create a set of stopwords.
    stop_words = set(stopwords.words('spanish')) 
    # Tokenize entry text.
    word_tokens = word_tokenize(sentence) 
    # Remove the stopwords from entry text.
    return ' '.join([w for w in word_tokens if not w in stop_words])

def stanford_lemma(text):
  doc = stanLemma(text)
  doc = ' '.join([word.lemma for sent in doc.sentences for word in sent.words  if word.upos != 'DET' and word.upos != 'PRON'])
  return doc.upper()

def convert_to_tokens(tempString, words_bag):
    tokens = []
    tempString = tempString.split(" ")
    for word in tempString:
        if(word in words_bag.keys()):
            tokens.append(words_bag[word])
    return tokens



def clean_tweet(tweet: str) -> str:
    tempString = removeLinks(tweet)
    tempString = removeHashtag(tempString)
    tempString = removeStopWords(tempString)
    tempString = removePunctuation(tempString)
    tempString = re.sub("\w+\d\w*|\w*\d\w+| \d", "", tempString)
    tempString = re.sub(" \w{1} ", " ", tempString)
    tempString = re.sub(" +NARCO\w", " NARCO", tempString)
    tempString = re.sub(" +IZQUIER\w", " IZQUIERDA", tempString)
    tempString = re.sub(" +IV(Á|A)N\w", " IVÁN", tempString)
    tempString = re.sub(" (J+|A+J|E+J)+ ", " RISA", tempString)
    tempString = re.sub(" +(URIBES\w+| *URIBIS\w+)", " URIBISTA", tempString)
    tempString = re.sub(" +(((Á|A)LVARO))? URIBE V(E|É)LEZ", " URIBE", tempString)
    tempString = re.sub(" +(PETRO\w+|PETRIS\w+)", " PETRISTA", tempString)
    tempString = re.sub(" POLOMB\w+", " CHISTE COLOMBIA", tempString)
    tempString = re.sub(" HIJUEP\w+", " INSULTO", tempString)
    tempString = re.sub(" BOBO", " INSULTO", tempString)
    tempString = re.sub(" HP", " INSULTO", tempString)
    if(len(tempString)>1):
        tempString = stanford_lemma(tempString)
        if(len(tempString) > 0):
            return tempString
    else:
        return None
    return None  

def text_to_tokens(tweet: str):
    default = np.zeros(24)
    tempString = clean_tweet(tweet)
    words_bag = load_obj("word_index")
    #Validate and drop empty text after processing
    if(len(tempString)>1):
        tempString = stanford_lemma(tempString)
        tempString = tempString.lower()
        if(len(tempString) > 0):
            tempString = convert_to_tokens(tempString, words_bag)
            return tempString
    else:
        return default   
    

def transform_prediction(value: float) -> float:
    if(value > threshold):
        return ((1/2) * ((value - threshold)/(1-threshold))**(1/2)) + (1/2)
    else:
        return (value**2)/(2*(threshold**2))
    
    
def predict_polarization(tweet: str):
    model = tf.keras.models.load_model("./models",compile=True)
    tweet = tweet.upper()
    input_lenght = model.get_layer(index = 0).get_config()['input_length']
    input_tweet = text_to_tokens(tweet)
    tweet_length = len(input_tweet)
    if(tweet_length <= input_lenght):
        for i in range (0, input_lenght - tweet_length):
            input_tweet.append(0)
    input_tweet = np.array(input_tweet)            
    input_tweet = input_tweet[:, np.newaxis]
    shape = np.shape(input_tweet)
    input_tweet = np.reshape(input_tweet, (shape[1], shape[0]))
    prediction = model.predict(input_tweet)
    prediction = prediction[0][0]
    return transform_prediction(prediction)

