#Pandas to handle csv and dataframes
import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'

#NLTK to handle text to be processed 
import re
from nltk.corpus import stopwords # Usado para eliminar las stopwords
# from nltk.tag import StanfordNERTagger # Usado para tagguear las palabras cómo entidades nombradas.

#Lemmatization
import stanza 
stanLemma = stanza.Pipeline(processors='tokenize,mwt,pos,lemma', lang='es', use_gpu=False)
from nltk.tokenize import word_tokenize # Word to tokens
# Word to tokens
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Import the csv which is going to be processed 
tweetsDF = pd.read_csv("/home/kodewill/PF/pf-twitter-data/Data/tweetsFirst.csv")


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
    tweet = re.sub('(#){1}(.)+', '',tweet)
    tweet = re.sub('(@){1}(.)+', '',tweet)
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
  return ' '.join([word.lemma for sent in doc.sentences for word in sent.words])

#Remove punctuation, hashtags, mentions, accent marks and stopwords.
for row, element in enumerate(tweetsDF['text']) :
    tempString = removeAccentMarks(tweetsDF['text'][row])
    tempString = removeLinks(tempString)
    tempString = removeHashtag(tempString)
    tempString = removeStopWords(tempString)
    tempString = removePunctuation(tempString)
    
    #Validate and drop empty text after processing
    if(len(tempString)>1):
        tweetsDF['text'].loc[row] = tempString
    else:
        tweetsDF['text'].drop([row])
    
# Lemmatization of dataframe
tweetsDF['text'] = tweetsDF['text'].apply(lambda x: stanford_lemma(x))

# Tokenize
tweets = tweetsDF['text'].tolist()
tokenizer = Tokenizer(num_words = 1000, oov_token="<OOV>")
tokenizer.fit_on_texts(tweets)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(tweets)
pad_seq = pad_sequences(sequences, padding='post')
print(pad_seq)

