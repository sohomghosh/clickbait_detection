
# coding: utf-8

# Quora Duplicate Question Detection (https://github.com/tgaddair/quora-duplicate-question-detector/blob/master/models/siamese_lstm.py)
# 
# Bi-directional Long Short Term Memory Network (LSTM) with Attention and Siamese Neural Network
# 
# https://github.com/sohomghosh/Solutions_of_Data_Science_Hackathons/blob/master/Kaggle/toxic_comment_v1/w2v_lstm_toxic_comment_keras_v1.py
# 
# https://github.com/sohomghosh/Solutions_of_Data_Science_Hackathons/blob/master/Kaggle/spooky_author_identification/xgboost_lstm_kaggle_spooky_nov17.py

# In[1]:


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


# In[2]:


da = pd.read_csv('/data/click_bait_detect/all_features_label_exceptImages_v3_with_pos_polarity_wmd_extrafeatures.csv')
for i in ['number_of_images', 'number_of_targetKeyWords']:
    if i in da.columns:
        del da[i]
da = da[~da.isin([np.nan, np.inf, -np.inf]).any(1)]
X = da.drop(['truthClass', 'truthClass_numeric'], axis =1)
y = da['truthClass_numeric']
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.33, random_state=42,stratify = y)


# In[3]:


da.shape


# In[4]:


y.dtype


# In[5]:


scl = StandardScaler()
X_train_scaled = scl.fit_transform(train_X) #Ideally should have been done on train_test
y_train = keras.utils.to_categorical(train_y, num_classes=2)
X_valid_scaled = scl.transform(valid_X)
y_valid = keras.utils.to_categorical(valid_y, num_classes=2)


# ## Model Type 1

# In[16]:


num_classes = 2

# create a simple 3 layer sequential neural net
model = Sequential()

#first parameter i.e. 50 in this case is number of neurons in hidden layer
#input_layer_neurons = input_dim = number of columns
model.add(Dense(50, input_dim=train_X.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(300, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')
#model.compile(loss='binary_crossentropy', optimizer='adam')

model.summary()


# In[17]:


model.fit(X_train_scaled, y=y_train, batch_size=64, 
          epochs=10, verbose=1, 
          validation_data=(X_valid_scaled, y_valid))


# In[18]:


X_train_scaled.shape


# In[19]:


train_preds = model.predict(np.array(X_train_scaled))[:,1]


# In[20]:


valid_preds = model.predict(np.array(X_valid_scaled))[:,1]


# In[21]:


train_y.shape


# In[22]:


train_preds.shape


# In[23]:


train_preds_class = model.predict_classes(np.array(X_train_scaled))


# In[24]:


valid_preds_class = model.predict_classes(np.array(X_valid_scaled))


# In[25]:


fig, ax = plt.subplots(figsize=(8,6))
print('\nTraining Accuracy:{}'.format(accuracy_score(train_y,train_preds_class)))
print('Mean Squared Error (MSE):{}'.format(mean_squared_error(train_y,train_preds)))
print('Training Confusion Matrix \n{}'.format(confusion_matrix(train_y,train_preds_class)))
print('Classification Report: \n{}'.format(classification_report(train_y,train_preds_class)))

fpr,tpr,threshold=roc_curve(train_y,train_preds)
ax.plot(fpr,tpr,label='Training AUC')
print('\nAUC:{}\n'.format(auc(fpr,tpr)))

print ('\nTest Accuracy:{}'.format(accuracy_score(valid_y,valid_preds_class)))
print('Mean Squared Error (MSE):{}'.format(mean_squared_error(valid_y,valid_preds)))
print ('Test Confusion Matrix \n{}'.format(confusion_matrix(valid_y,valid_preds_class)))
print('Classification Report: \n{}'.format(classification_report(valid_y,valid_preds_class)))
fpr,tpr,threshold=roc_curve(valid_y,valid_preds)
print ('\nAUC:{}\n'.format(auc(fpr,tpr)))
ax.plot(fpr,tpr,label='Test AUC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
ax.legend(loc='best')
plt.show()


# In[50]:


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


# In[100]:


raw_data_cleaned = pd.read_csv('/data/click_bait_detect/data_with_cleaned_splitted_text_new.csv')


# In[101]:


XX = raw_data_cleaned['postText']
yy = raw_data_cleaned['truthClass'].replace({'no-clickbait':0, 'clickbait':1})
xtrain, xvalid, ytrain, yvalid = train_test_split(XX, yy, 
                                                  stratify=yy, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)


# In[102]:


X.shape


# In[103]:


xtrain.shape


# In[104]:


# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 70
token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)


# In[105]:


len(xtrain_seq)


# In[107]:


# we need to binarize the labels for the neural net
ytrain_enc = np_utils.to_categorical(ytrain)
yvalid_enc = np_utils.to_categorical(yvalid)

# zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index


# In[108]:


xtrain_pad.shape


# In[109]:


sentences_split = [eval(i) for i in raw_data_cleaned['postText_cleaned_splitted']]


# In[111]:


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


# ## Model Type 2 - LSTM Model

# In[112]:


# A simple LSTM with two dense layers
model.reset_states()
model = Sequential()
model.add(Embedding(vocab_size,
                     word_vec_dim,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[130]:


word_vec_dim


# In[129]:


vocab_size


# In[131]:


max_len


# In[113]:


model.summary()


# In[114]:


ytrain_enc.shape


# In[115]:


xtrain_pad.shape


# In[117]:


# Fit the model with early stopping callback : Early stoping prevents overfiting
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
#checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=10, verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks = [earlystop]) #callbacks = [earlystop, checkpoint]

##Without early stopping
##model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=100, verbose=1, validation_data=(xvalid_pad, yvalid_enc))


# In[118]:


train_preds_class = model.predict_classes(np.array(xtrain_pad))
valid_preds_class = model.predict_classes(np.array(xvalid_pad))


# In[119]:


train_preds = model.predict(np.array(xtrain_pad))[:,1]
valid_preds = model.predict(np.array(xvalid_pad))[:,1]


# In[124]:


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


# ## Model Type 3 - Bi-directional LSTM

# In[125]:


model.reset_states()
model = Sequential()
model.add(Embedding(vocab_size,
                     word_vec_dim,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(350, dropout=0.3, recurrent_dropout=0.3)))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=10, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])


# In[126]:


train_preds_class = model.predict_classes(np.array(xtrain_pad))
valid_preds_class = model.predict_classes(np.array(xvalid_pad))
train_preds = model.predict(np.array(xtrain_pad))[:,1]
valid_preds = model.predict(np.array(xvalid_pad))[:,1]
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


# ## Model Type 4 - GRU

# In[127]:


model.reset_states()
model = Sequential()
model.add(Embedding(vocab_size,
                     word_vec_dim,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
model.add(GRU(300, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(xtrain_pad, y=ytrain_enc, batch_size=512, epochs=10, 
          verbose=1, validation_data=(xvalid_pad, yvalid_enc), callbacks=[earlystop])


# In[128]:


train_preds_class = model.predict_classes(np.array(xtrain_pad))
valid_preds_class = model.predict_classes(np.array(xvalid_pad))
train_preds = model.predict(np.array(xtrain_pad))[:,1]
valid_preds = model.predict(np.array(xvalid_pad))[:,1]
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

