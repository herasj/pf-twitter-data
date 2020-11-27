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
from keras import layers
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K

#Tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings


#F1 metric for unbalanced binary classes
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

#Save object
import pickle 
def save_obj(obj, name ):
    with open('./predict_scripts/'+ name + '.pkl', 'wb') as f:
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

def tf_idf(corpus, numWords: int):
    vectorizer = TfidfVectorizer(max_features=numWords)
    X = vectorizer.fit_transform(corpus)
    X = vectorizer.get_feature_names()
    word_index = {word: index for index, word in enumerate(X)}
    corpus_tokens = [tweet.split(" ") for tweet in corpus]
    sequences = [[word_index[word.lower()] for word in innerList if word.lower() in word_index] for innerList in corpus_tokens]
    maxLen = max([len(tweet) for tweet in sequences])
    sequences = [sequence + [0]*(maxLen - len(sequence)) for sequence in sequences]    
    return word_index, sequences
        
def split_dataset(X, y):
    #Class weights for unbalanced data set
    total = np.shape(X)[0]
    pos = sum(y)
    neg = total - pos
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0    
    class_weight = {0: weight_for_0, 1: weight_for_1}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    return X_train, X_test, y_train, y_test, class_weight


#Simple generic embedding model
def EmbeddingNN(X_train, X_test, y_train, y_test, class_weights, num_units, input_shape, dropoutFirst, dropout, lr, actOne, actTwo, vocab_size, embedding_dim, num_epochs):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_shape),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(dropoutFirst),
        tf.keras.layers.Dense(num_units, activation=actOne),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation=actTwo)
    ])
    model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate = lr), 
                  metrics= ['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test),
                        class_weight=class_weights, verbose=2, batch_size=44)
    tf.keras.models.save_model(model, filepath)
    loss, accuracy = model.evaluate(X_test, y_test)
    return model, accuracy


def plot_conf_matrix(conf_matrix, conf_matrix_norm, title):    
    #Plot confusion matrix
    class_names = ['NO POLARIZANTE', 'POLARIZANTE']
    import seaborn as sn
    import matplotlib.pyplot as plt
    #Conf matrix
    df_cm = pd.DataFrame(conf_matrix, index = class_names,
                      columns = class_names)
    df_cm.index.name = 'ACTUAL'
    df_cm.columns.name = 'PREDICTED'
    plt.figure(figsize = (10,7))
    plt.title(title, fontsize =20)
    plt.yticks(np.arange(7)+0.5, class_names, rotation=0, fontsize="15", va="center")
    plt.xticks(np.arange(7)+0.5, class_names, rotation=0, fontsize="15", va="center")
    sn.heatmap(df_cm, annot=True)
    
    #conf matrix normalized
    df_cm = pd.DataFrame(conf_matrix_norm, index = class_names,
                      columns = class_names)
    plt.figure(figsize = (10,7))
    df_cm.index.name = 'ACTUAL'
    df_cm.columns.name = 'PREDICTED'
    plt.title(title + " NORMALIZED", fontsize =20)
    plt.yticks(np.arange(7)+0.5, class_names, rotation=0, fontsize="15", va="center")
    plt.xticks(np.arange(7)+0.5, class_names, rotation=0, fontsize="15", va="center")
    sn.heatmap(df_cm, annot=True)


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


#Import the csv which is going to be processed 
tweetsDF = pd.read_csv("/home/kodewill/PF/pf-twitter-data/Data/polarization.csv")
filepath = '/home/kodewill/PF/pf-twitter-data/models/'

finalTweets = {}
delRow = []
for row, element in enumerate(tweetsDF.iterrows()) :
    tempString = clean_tweet(tweetsDF['text'][row])
    #Validate and drop empty text after processing
    if(tempString):
        # tempSentiment = [tweetsDF[sentiment[0]][row], tweetsDF[sentiment[1]][row], tweetsDF[sentiment[2]][row], tweetsDF[sentiment[3]][row], sentimentMap[tweetsDF[sentiment[4]][row]]]
        # sentimentsRows.append(tempSentiment)
        finalTweets[row] = {'polarization': tweetsDF['polarization'][row], 'tweet': tempString}
    else:
        delRow.append(row)

#TF-IDF
tweetsDF = tweetsDF.drop(delRow) 
tweets = [innerDict['tweet'] for row, innerDict in finalTweets.items()]
vocab_size = 400
embedding_dim = 128
word_index, tokenizedTweets = tf_idf(tweets, vocab_size)


# # Tokenize
# vocab_size = 1000
# embedding_dim = 256
# max_length = 500
# padding_type='post'
# oov_tok = "<OOV>"
# tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
# tokenizer.fit_on_texts(tweets)
# word_index = tokenizer.word_index
# word_index = {word: index for word, index in word_index.items() if index <= 1000}
# sequences = tokenizer.texts_to_sequences(tweets)
# pad_seq = pad_sequences(sequences, padding='post')
# tweets = pd.DataFrame(pad_seq)
#Add aditional information (Amazon sentiment analysis)


#Embedding
X = np.array(tokenizedTweets)
y = [1 if innerDict['polarization'] and row not in delRow else (0 if row not in delRow else None) for row, innerDict in finalTweets.items()]
y = [value for value in y if value != None]
y = np.array(y)
X_train, X_test, y_train, y_test, class_weights = split_dataset(X,y)
num_units = 16
actOne = 'sigmoid'
actTwo = 'sigmoid'
input_shape = np.shape(X)[1]
dropoutFirst = 0.7
dropout = 0.5
lr = 0.002
num_epochs = 50

# #plot pca
# import plotly.express as px
# from plotly.offline import plot
# from sklearn.decomposition import PCA

# pca = PCA(n_components=3)
# components = pca.fit_transform(X)

# total_var = pca.explained_variance_ratio_.sum() * 100

# fig = px.scatter_3d(
#     components, x=0, y=1, z=2, color=y,
#     title=f'Total Explained Variance: {total_var:.2f}%'
# )
# plot(fig)

#Embedding
model, accuracy = EmbeddingNN(X_train, X_test, y_train, y_test, class_weights, 
                              num_units, input_shape, dropout, dropoutFirst, lr, actOne, 
                              actTwo, vocab_size, embedding_dim, num_epochs)


#Find best threshold
from sklearn.metrics import roc_curve, precision_recall_curve
from matplotlib import pyplot
from numpy import sqrt
from numpy import argmax#plot pca
import plotly.express as px
from plotly.offline import plot
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
components = pca.fit_transform(X)

total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=y,
    title=f'Total Explained Variance: {total_var:.2f}%'
)
plot(fig)

# predict probabilities
yhat = model.predict_proba(X_test)
# calculate roc curves
fpr, tpr, thresholdsROC = roc_curve(y_test, yhat)
# calculate the g-mean for each threshold
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ixROC = argmax(gmeans)
print('Best Threshold ROC=%f, G-Mean=%.3f' % (thresholdsROC[ixROC], gmeans[ixROC]))
# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
pyplot.scatter(fpr[ixROC], tpr[ixROC], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()

# # calculate pr curves
# precision, recall, thresholdsPR = precision_recall_curve(y_test, yhat)
# # convert to f score
# fscore = (2 * precision * recall) / (precision + recall)
# # locate the index of the largest f score
# ixPR = sorted(range(len(precision)), key=lambda i: precision[i], reverse=True)[:6]
# ixPR = list(fscore).index(max(fscore[ixPR]))
# print('Best Threshold PR=%f, F-Score=%.3f' % (thresholdsPR[ixPR], fscore[ixPR]))
# # plot the roc curve for the model, random_state=0
# no_skill = len(y_test[y_test==1]) / len(y_test)
# pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
# pyplot.plot(recall, precision, marker='.', label='Logistic')
# pyplot.scatter(recall[ixPR], precision[ixPR], marker='o', color='black', label='Best')
# # axis labels
# pyplot.xlabel('Recall')
# pyplot.ylabel('Precision')
# pyplot.legend()
# # show the plot
# pyplot.show()


#Confusion matrix 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# ROC CURVE 
y_pred = model.predict(X_test)
y_pred = np.array([1 if row > thresholdsROC[ixROC] else 0 for row in y_pred])
y_test = np.array(y_test)

conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_norm = confusion_matrix(y_test, y_pred, normalize = 'true')
plot_conf_matrix(conf_matrix, conf_matrix_norm, "ROC CONFUSION MATRIX")

#PR CURVE 
# y_pred = model.predict(X_test)
# y_pred = np.array([1 if row > thresholdsPR[ixPR] else 0 for row in y_pred])
# y_test = np.array(y_test)

# conf_matrix = confusion_matrix(y_test, y_pred)
# conf_matrix_norm = confusion_matrix(y_test, y_pred, normalize = 'true')
# plot_conf_matrix(conf_matrix, conf_matrix_norm, "PR CONFUSION MATRIX")

