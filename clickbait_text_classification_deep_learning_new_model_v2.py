
# coding: utf-8

# In[2]:


import pandas as pd
import string
import numpy as np
import re
#from gensim.models import word2vec
from gensim.models import Word2Vec
import nltk
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,cross_val_score
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tag import pos_tag
from collections import Counter
from sklearn.preprocessing import StandardScaler
import keras
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Sequential
from nltk.data import load
from keras.layers import Dense, Activation
from numpy.random import random, normal
from sklearn.metrics import accuracy_score,roc_curve,classification_report,confusion_matrix,precision_recall_curve,auc, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


# In[3]:


from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# In[4]:


da = pd.read_csv('/data/click_bait_detect/all_features_label_exceptImages_v3_with_pos_polarity_wmd_extrafeatures.csv')
for i in ['number_of_images', 'number_of_targetKeyWords']:
    if i in da.columns:
        del da[i]
da = da[~da.isin([np.nan, np.inf, -np.inf]).any(1)]


# In[37]:


raw_data_cleaned = pd.read_csv('/data/click_bait_detect/data_with_cleaned_splitted_text_new.csv')


# In[38]:


raw_data_cleaned.columns


# In[39]:


XX = raw_data_cleaned[['postText_cleaned_splitted', 'targetTitle_cleaned_splitted', 'targetKeywords_cleaned_splitted', 'targetDescription_cleaned_splitted', 'targetParagraphs_cleaned_splitted']]
yy = raw_data_cleaned['truthClass'].replace({'no-clickbait':0, 'clickbait':1})
xtrain, xvalid, ytrain, yvalid = train_test_split(XX, yy, 
                                                  stratify=yy, 
                                                  random_state=42, 
                                                  test_size=0.33, shuffle=True)


# In[40]:


xtrain.head()


# In[41]:


xvalid.head()


# In[42]:


all_texts = []
for col in ['postText_cleaned_splitted', 'targetTitle_cleaned_splitted', 'targetKeywords_cleaned_splitted', 'targetDescription_cleaned_splitted', 'targetParagraphs_cleaned_splitted']:
    all_texts = all_texts + list(xtrain[col].apply(lambda x : ' '.join(eval(str(x))))) 
    all_texts = all_texts + list(xvalid[col].apply(lambda x : ' '.join(eval(str(x))))) 


# In[43]:


all_texts[:5]


# In[44]:


# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 140
token.fit_on_texts(all_texts)


# In[45]:


xtrain_postText_seq = token.texts_to_sequences(xtrain['postText_cleaned_splitted'].apply(lambda x : ' '.join(eval(str(x)))))
xvalid_postText_seq = token.texts_to_sequences(xvalid['postText_cleaned_splitted'].apply(lambda x : ' '.join(eval(str(x)))))

xtrain_targetTitle_seq = token.texts_to_sequences(xtrain['targetTitle_cleaned_splitted'].apply(lambda x : ' '.join(eval(str(x)))))
xvalid_targetTitle_seq = token.texts_to_sequences(xvalid['targetTitle_cleaned_splitted'].apply(lambda x : ' '.join(eval(str(x)))))

xtrain_targetKeywords_seq = token.texts_to_sequences(xtrain['targetKeywords_cleaned_splitted'].apply(lambda x : ' '.join(eval(str(x)))))
xvalid_targetKeywords_seq = token.texts_to_sequences(xvalid['targetKeywords_cleaned_splitted'].apply(lambda x : ' '.join(eval(str(x)))))

xtrain_targetDescription_seq = token.texts_to_sequences(xtrain['targetDescription_cleaned_splitted'].apply(lambda x : ' '.join(eval(str(x)))))
xvalid_targetDescription_seq = token.texts_to_sequences(xvalid['targetDescription_cleaned_splitted'].apply(lambda x : ' '.join(eval(str(x)))))

xtrain_targetParagraphs_seq = token.texts_to_sequences(xtrain['targetParagraphs_cleaned_splitted'].apply(lambda x : ' '.join(eval(str(x)))))
xvalid_targetParagraphs_seq = token.texts_to_sequences(xvalid['targetParagraphs_cleaned_splitted'].apply(lambda x : ' '.join(eval(str(x)))))


# In[46]:


# we need to binarize the labels for the neural net
ytrain_enc = np_utils.to_categorical(ytrain)
yvalid_enc = np_utils.to_categorical(yvalid)


# In[47]:


# zero pad the sequences
xtrain_postText_pad = sequence.pad_sequences(xtrain_postText_seq, maxlen=max_len)
xvalid_postText_pad = sequence.pad_sequences(xvalid_postText_seq, maxlen=max_len)

xtrain_targetTitle_pad = sequence.pad_sequences(xtrain_targetTitle_seq, maxlen=max_len)
xvalid_targetTitle_pad = sequence.pad_sequences(xvalid_targetTitle_seq, maxlen=max_len)

xtrain_targetKeywords_pad = sequence.pad_sequences(xtrain_targetKeywords_seq, maxlen=max_len)
xvalid_targetKeywords_pad = sequence.pad_sequences(xvalid_targetKeywords_seq, maxlen=max_len)

xtrain_targetDescription_pad = sequence.pad_sequences(xtrain_targetDescription_seq, maxlen=max_len)
xvalid_targetDescription_pad = sequence.pad_sequences(xvalid_targetDescription_seq, maxlen=max_len)

xtrain_targetParagraphs_pad = sequence.pad_sequences(xtrain_targetParagraphs_seq, maxlen=max_len)
xvalid_targetParagraphs_pad = sequence.pad_sequences(xvalid_targetParagraphs_seq, maxlen=max_len)

word_index = token.word_index


# In[48]:


xtrain_targetParagraphs_pad.shape


# In[49]:


sentences_split = [sent.split() for sent in all_texts]


# In[50]:


# create an embedding matrix for the words we have in the dataset
vocab_size = len(word_index) + 1
word_vec_dim = 300
embedding_matrix = np.zeros((vocab_size, word_vec_dim))
word2vec = Word2Vec(sentences_split, size=word_vec_dim, min_count =1, window=3, workers =-1,sample=1e-5)

for word, i in tqdm(word_index.items()):
    try:
        embedding_matrix[i] = word2vec.wv[word] #FOR GLOVE# embeddings_index.get(word)
    except KeyError:
        pass


# # Model LSTM

# In[51]:


num_classes = 2


# In[52]:


max_len


# In[53]:


num_features = 5 #As we have 5 features : 'postText_cleaned_splitted', 'targetTitle_cleaned_splitted', 'targetKeywords_cleaned_splitted', 'targetDescription_cleaned_splitted', 'targetParagraphs_cleaned_splitted'


# In[ ]:


# A simple LSTM with two dense layers
model.reset_states()
model = Sequential()
model.add(Embedding(vocab_size,
                     word_vec_dim,
                     weights=[embedding_matrix],
                     input_length=max_len*num_features,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(300, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(400, activation='relu'))
model.add(Dropout(0.8))

#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.8))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()


# In[ ]:


earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model.fit(np.hstack((xtrain_postText_pad, xtrain_targetTitle_pad, xtrain_targetKeywords_pad, xtrain_targetDescription_pad, xtrain_targetParagraphs_pad)), y=ytrain_enc, batch_size=64, 
          epochs=5, verbose=1, 
          validation_data=(np.hstack((xvalid_postText_pad, xvalid_targetTitle_pad, xvalid_targetKeywords_pad, xvalid_targetDescription_pad, xvalid_targetParagraphs_pad)), yvalid_enc))


# In[ ]:


model.save_weights("/data/click_bait_detect/text_classification_model_lstm_5_features.h5")


# In[47]:


np.hstack((xtrain_postText_pad, xtrain_targetTitle_pad, xtrain_targetKeywords_pad, xtrain_targetDescription_pad, xtrain_targetParagraphs_pad))


# In[43]:


np.hstack((xvalid_postText_pad, xvalid_targetTitle_pad, xvalid_targetKeywords_pad, xvalid_targetDescription_pad, xvalid_targetParagraphs_pad))


# In[ ]:


train_preds_class = model.predict_classes(np.array(np.hstack((xtrain_postText_pad, xtrain_targetTitle_pad, xtrain_targetKeywords_pad, xtrain_targetDescription_pad, xtrain_targetParagraphs_pad))))
valid_preds_class = model.predict_classes(np.array(np.hstack((xvalid_postText_pad, xvalid_targetTitle_pad, xvalid_targetKeywords_pad, xvalid_targetDescription_pad, xvalid_targetParagraphs_pad))))


# In[ ]:


train_preds = model.predict(np.array(np.hstack((xtrain_postText_pad, xtrain_targetTitle_pad, xtrain_targetKeywords_pad, xtrain_targetDescription_pad, xtrain_targetParagraphs_pad))))[:,1]
valid_preds = model.predict(np.array(np.hstack((xvalid_postText_pad, xvalid_targetTitle_pad, xvalid_targetKeywords_pad, xvalid_targetDescription_pad, xvalid_targetParagraphs_pad))))[:,1]


# In[60]:


fig, ax = plt.subplots(figsize=(8,6))
print('\nTraining Accuracy:{}'.format(accuracy_score(ytrain,train_preds_class)))
print('Mean Squared Error (MSE):{}'.format(mean_squared_error(ytrain,train_preds)))
print('Training Confusion Matrix \n{}'.format(confusion_matrix(ytrain,train_preds_class)))
print('Classification Report: \n{}'.format(classification_report(ytrain,train_preds_class)))

fpr,tpr,threshold=roc_curve(ytrain,train_preds)
ax.plot(fpr,tpr,label='Training AUC')
print('\nAUC:{}\n'.format(auc(fpr,tpr)))

print ('\nTest Accuracy:{}'.format(accuracy_score(yvalid,valid_preds_class)))
print('Mean Squared Error (MSE):{}'.format(mean_squared_error(yvalid,valid_preds)))
print ('Test Confusion Matrix \n{}'.format(confusion_matrix(yvalid,valid_preds_class)))
print('Classification Report: \n{}'.format(classification_report(yvalid,valid_preds_class)))
fpr,tpr,threshold=roc_curve(yvalid,valid_preds)
print ('\nAUC:{}\n'.format(auc(fpr,tpr)))
ax.plot(fpr,tpr,label='Test AUC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
ax.legend(loc='best')
plt.show()


# # Model LSTM

# In[27]:


# A simple LSTM with two dense layers
model.reset_states()
model = Sequential()
model.add(Embedding(vocab_size,
                     word_vec_dim,
                     weights=[embedding_matrix],
                     input_length=max_len*num_features,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(800, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(500, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()


# In[ ]:


earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model.fit(np.hstack((xtrain_postText_pad, xtrain_targetTitle_pad, xtrain_targetKeywords_pad, xtrain_targetDescription_pad, xtrain_targetParagraphs_pad)), y=ytrain_enc, batch_size=64, 
          epochs=5, verbose=1, 
          validation_data=(np.hstack((xvalid_postText_pad, xvalid_targetTitle_pad, xvalid_targetKeywords_pad, xvalid_targetDescription_pad, xvalid_targetParagraphs_pad)), yvalid_enc))


# In[ ]:


model.save_weights("/data/click_bait_detect/text_classification_model_lstm_5_features_really_deep.h5")


# In[ ]:


train_preds_class = model.predict_classes(np.array(np.hstack((xtrain_postText_pad, xtrain_targetTitle_pad, xtrain_targetKeywords_pad, xtrain_targetDescription_pad, xtrain_targetParagraphs_pad))))
valid_preds_class = model.predict_classes(np.array(np.hstack((xvalid_postText_pad, xvalid_targetTitle_pad, xvalid_targetKeywords_pad, xvalid_targetDescription_pad, xvalid_targetParagraphs_pad))))


# In[ ]:


train_preds = model.predict(np.array(np.hstack((xtrain_postText_pad, xtrain_targetTitle_pad, xtrain_targetKeywords_pad, xtrain_targetDescription_pad, xtrain_targetParagraphs_pad))))[:,1]
valid_preds = model.predict(np.array(np.hstack((xvalid_postText_pad, xvalid_targetTitle_pad, xvalid_targetKeywords_pad, xvalid_targetDescription_pad, xvalid_targetParagraphs_pad))))[:,1]


# In[ ]:


fig, ax = plt.subplots(figsize=(8,6))
print('\nTraining Accuracy:{}'.format(accuracy_score(ytrain,train_preds_class)))
print('Mean Squared Error (MSE):{}'.format(mean_squared_error(ytrain,train_preds)))
print('Training Confusion Matrix \n{}'.format(confusion_matrix(ytrain,train_preds_class)))
print('Classification Report: \n{}'.format(classification_report(ytrain,train_preds_class)))

fpr,tpr,threshold=roc_curve(ytrain,train_preds)
ax.plot(fpr,tpr,label='Training AUC')
print('\nAUC:{}\n'.format(auc(fpr,tpr)))

print ('\nTest Accuracy:{}'.format(accuracy_score(yvalid,valid_preds_class)))
print('Mean Squared Error (MSE):{}'.format(mean_squared_error(yvalid,valid_preds)))
print ('Test Confusion Matrix \n{}'.format(confusion_matrix(yvalid,valid_preds_class)))
print('Classification Report: \n{}'.format(classification_report(yvalid,valid_preds_class)))
fpr,tpr,threshold=roc_curve(yvalid,valid_preds)
print ('\nAUC:{}\n'.format(auc(fpr,tpr)))
ax.plot(fpr,tpr,label='Test AUC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
ax.legend(loc='best')
plt.show()

