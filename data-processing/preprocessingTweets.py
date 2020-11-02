#Pandas and numpy to handle csv and dataframes and other data structures
import numpy as np
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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Prepare data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#word2vec
import warnings   
warnings.filterwarnings(action = 'ignore')   
import gensim 
from gensim.models import Word2Vec 
from gensim.test.utils import get_tmpfile

#DNN
from keras.models import Sequential
from keras import layers

#Save object
import pickle 
def save_obj(obj, name ):
    with open('predict_scripts/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 0)




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

def tokenizeTweet(tweet: str)->list:
    # tweet = re.sub(r'[^a-zA-Z0-9\s]', ' ', tweet)
    tokens = [token for token in tweet.split(" ") if token != ""]
    return tokens

#Function to add an array with extra information (sentiment scores, replies, fav and retweets)
def addInfoArray(infoRow):
    array = np.zeros(100)
    array[0] = infoRow['favorites']
    array[1] = infoRow['replies']
    array[2] = infoRow['retweets']
    array[3] = infoRow['sentimentScore.mixed']
    array[4] = infoRow['sentimentScore.neutral']
    array[5] = infoRow['sentimentScore.positive']
    array[6] = infoRow['sentimentScore.negative']
    return array 

def addInfo(infoRow, frameRow):
    index = len(frameRow)
    tempRow = frameRow
    tempRow[index + 0] = infoRow['favorites']
    tempRow[index + 1] = infoRow['replies']
    tempRow[index + 2] = infoRow['retweets']
    tempRow[index + 3] = infoRow['sentimentScore.mixed']
    tempRow[index + 4] = infoRow['sentimentScore.neutral']
    tempRow[index + 5] = infoRow['sentimentScore.positive']
    tempRow[index + 6] = infoRow['sentimentScore.negative']
    return tempRow 

def split_dataset(X, y):
    #Class weights for unbalanced data set
    total = np.size(X)
    pos = sum(y)
    neg = total - pos
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0    
    class_weight = {0: weight_for_0, 1: weight_for_1}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.26)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    return X_train, X_test, y_train, y_test, class_weight

#Simple generic embedding model
def EmbeddingNN(X_train, X_test, y_train, y_test, class_weights, num_units, input_shape, dropout, lr, actOne, actTwo, vocab_size, embedding_dim, num_epochs):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_shape),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(num_units, activation=actOne),
        tf.keras.layers.Dense(1, activation=actTwo)
    ])
    model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate = lr), 
                  metrics=['accuracy'])
    model.summary()
    model.save(
        filepath,
        overwrite=False,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        )
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test), verbose=2)

    loss, accuracy = model.evaluate(X_test, y_test)
    
    return accuracy

#First Model for (Word2Vec)

def CNN(X_train, X_test, y_train, y_test, class_weights, num_units, input_shape, dropout, lr, actOne, actTwo, actThree):
    model = Sequential()
    model.add(layers.Conv1D(num_units, 1, activation=actOne, input_shape=input_shape))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation=actTwo))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(6, activation=actThree))
    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    model.summary()
    model.save(
        filepath,
        overwrite=False,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        )

    loss, accuracy = model.evaluate(X_test, y_test)
    
    return accuracy

#Import the csv which is going to be processed 
tweetsDF = pd.read_csv("/home/kodewill/PF/pf-twitter-data/Data/tweetsFirst.csv")
filepath = '/home/kodewill/PF/pf-twitter-data/models/'

finalTweets = {}
delRow = []
# sentiment = ['sentimentScore.mixed', 'sentimentScore.negative', 'sentimentScore.neutral', 'sentimentScore.positive', 'sentimentScore.predominant']
# sentimentMap = {'NEGATIVE': '-1', 'MIXED': '-0.5', 'NEUTRAL': '0', 'POSITIVE': '1'}
# sentimentsRows = []
#Remove punctuation, hashtags, mentions, accent marks and stopwords.
for row, element in enumerate(tweetsDF.iterrows()) :
    tempString = removeLinks(tweetsDF['text'][row])
    tempString = removeHashtag(tempString)
    tempString = removeStopWords(tempString)
    tempString = removePunctuation(tempString)
    
    #Validate and drop empty text after processing
    if(len(tempString)>1):
        tempString = stanford_lemma(tempString)
        if(len(tempString) > 0):
            # tempSentiment = [tweetsDF[sentiment[0]][row], tweetsDF[sentiment[1]][row], tweetsDF[sentiment[2]][row], tweetsDF[sentiment[3]][row], sentimentMap[tweetsDF[sentiment[4]][row]]]
            # sentimentsRows.append(tempSentiment)
            finalTweets[row] = {'political': tweetsDF['political'][row], 'tweet': tempString}
    else:
        delRow.append(row)

tweetsDF = tweetsDF.drop(delRow) 

# Tokenize
vocab_size = 2000
embedding_dim = 64
max_length = 500
padding_type='post'
oov_tok = "<OOV>"

tweets = [innerDict['tweet'] for row, innerDict in finalTweets.items()]
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(tweets)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(tweets)
pad_seq = pad_sequences(sequences, padding='post')
tweets = pd.DataFrame(pad_seq)
#Add aditional information (Amazon sentiment analysis)
    

    

#word2vec
tokenizedTweets = {index: tokenizeTweet(innerDict['tweet']) for index, innerDict in finalTweets.items()}
tokenizedTweets = list(tokenizedTweets.values())
path = get_tmpfile("word2vec.model")
model = Word2Vec(tokenizedTweets, window=5, min_count=3, workers=4)
model.save("word2vec.model")
wordVectors = model.wv

tweetVectors = {}
numExcep = 0
padder = np.zeros(100)
maxim = 0
count = 0

for index, tokenList in enumerate(tokenizedTweets):
    
    vecList = []
    for word in tokenList:
        try:
            tempVec = np.asarray(wordVectors.word_vec(word))
            vecList.append(tempVec)
        except:
            numExcep += 1
    
    #Encontrar el número máximo de palabras en un tweet            
    if(len(vecList) > maxim):
        maxim = len(vecList)
    
    if(len(vecList)  > 0):        
        tweetVectors[index] = np.asarray(vecList)

#Add a padder to have all arrays with the same shape
for index, palabras in tweetVectors.items():
    tempLen = len(tweetVectors[index])
    tempArray = addInfoArray(tweetsDF.iloc[index])
    tweetVectors[index] = np.concatenate((tweetVectors[index], [tempArray]), axis=0)
    if(tempLen < maxim and tempLen > 0):
        for i in range (0, maxim - tempLen):
            tweetVectors[index] = np.concatenate((tweetVectors[index], [padder]), axis=0)
tweetsVectorsList = []

for tweet in tweetVectors.values() :
    tweetsVectorsList.append(tweet)

tweetsVectorsList = np.array(tweetsVectorsList)

#Embedding model call

X = np.array(tweets)
y = [1 if innerDict['political'] else 0 for row, innerDict in finalTweets.items()]
y = np.array(y)
X_train, X_test, y_train, y_test, class_weights = split_dataset(X,y)
num_units = 32
actOne = 'relu'
actTwo = 'sigmoid'
input_shape = np.shape(X)[1]
dropout = None
lr = 0.005
num_epochs = 50

#Confusion matrix 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
y_pred = np.array([1 if row > 0.5 else 0 for row in y_pred])
y_test = np.array(y_test)

conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_norm = confusion_matrix(y_test, y_pred, normalize = 'true')

#Plot confusion matrix
class_names = ['NO POLÍTICO', 'POLÍTICO']
import seaborn as sn
import matplotlib.pyplot as plt

#Conf matrix
df_cm = pd.DataFrame(conf_matrix, index = class_names,
                  columns = class_names)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

#conf matrix normalized
df_cm = pd.DataFrame(conf_matrix_norm, index = class_names,
                  columns = class_names)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

