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

def text_to_tokens(tweet: str):
    default = np.zeros(37)
    tempString = removeLinks(tweet)
    tempString = removeHashtag(tempString)
    tempString = removeStopWords(tempString)
    tempString = removePunctuation(tempString)
    words_bag = load_obj("word_bag")
    #Validate and drop empty text after processing
    if(len(tempString)>1):
        tempString = stanford_lemma(tempString)
        tempString = tempString.lower()
        if(len(tempString) > 0):
            tempString = convert_to_tokens(tempString, words_bag)
            return tempString
    else:
        return default
    
    
def predict_class(tweet: str):
    model = keras.models.load_model("./models/")
    input_lenght = model.get_layer(index = 0).get_config()['input_length']
    input_tweet = text_to_tokens(tweet)
    tweet_length = len(input_tweet)
    if(tweet_length <= input_lenght):
        for i in range (0, input_lenght - tweet_length):
            input_tweet.append(0)
    input_tweet = np.array(input_tweet)            
    input_tweet = input_tweet[:, np.newaxis]
    return model.predict(input_tweet)