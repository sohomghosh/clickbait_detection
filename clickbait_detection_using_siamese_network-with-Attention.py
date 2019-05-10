
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
get_ipython().magic('matplotlib inline')


# In[3]:


from keras.models import Sequential
import time
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
import seaborn as sns
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import *
from keras.engine.topology import Layer
from keras import backend as K
from keras.regularizers import l2
K.set_image_data_format('channels_last')
import cv2
import os
from skimage import io
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
#from fr_utils import *
#from inception_blocks_v2 import *
import numpy.random as rng
from sklearn.utils import shuffle


# In[4]:


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


# In[6]:


import os
import re
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate, Lambda
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate


# In[23]:


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):#https://www.kaggle.com/fareise/multi-head-self-attention-for-text-classification
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


# In[7]:


raw_data_cleaned = pd.read_csv('/data/click_bait_detect/data_with_cleaned_splitted_text_new.csv')
XX = raw_data_cleaned[['postText_cleaned_splitted', 'targetTitle_cleaned_splitted', 'targetKeywords_cleaned_splitted', 'targetDescription_cleaned_splitted', 'targetParagraphs_cleaned_splitted']]
yy = raw_data_cleaned['truthClass'].replace({'no-clickbait':0, 'clickbait':1})
xtrain, xvalid, ytrain, yvalid = train_test_split(XX, yy, 
                                                  stratify=yy, 
                                                  random_state=42, 
                                                  test_size=0.33, shuffle=True)


# In[8]:


all_texts = []
for col in ['postText_cleaned_splitted', 'targetTitle_cleaned_splitted', 'targetKeywords_cleaned_splitted', 'targetDescription_cleaned_splitted', 'targetParagraphs_cleaned_splitted']:
    all_texts = all_texts + list(xtrain[col].apply(lambda x : ' '.join(eval(str(x))))) 
    all_texts = all_texts + list(xvalid[col].apply(lambda x : ' '.join(eval(str(x))))) 


# In[9]:


token = text.Tokenizer(num_words=None)
max_len = 140 #max_len is maximum lemgth of the sequence
token.fit_on_texts(all_texts)


# In[10]:


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


# In[11]:


# we need to binarize the labels for the neural net
#ytrain_enc = np_utils.to_categorical(ytrain)
#yvalid_enc = np_utils.to_categorical(yvalid)


# In[12]:


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


# In[13]:


xtrain_targetParagraphs_pad.shape


# In[14]:


sentences_split = [sent.split() for sent in all_texts]


# In[15]:


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


# In[42]:


input_shape = (max_len,)#max_len = length of sequence = number of columns
left_input = Input(input_shape)
right_input = Input(input_shape)

model = Sequential()

model.add(Embedding(vocab_size,
                     word_vec_dim,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))

#model.add(Dense(500, activation='relu'))
#model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Bidirectional(LSTM(300, return_sequences=True)))

model.add(Attention(max_len))

encoded_l = model(left_input)
encoded_r = model(right_input)
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_l, encoded_r])
prediction = Dense(1,activation='sigmoid')(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)


# In[43]:


model.summary()


# In[44]:


siamese_net.summary()


# In[45]:


siamese_net.compile(loss='binary_crossentropy', optimizer='adam')


# In[ ]:


siamese_net.fit([xtrain_postText_pad, xtrain_targetDescription_pad],y=ytrain,batch_size=64, 
          epochs=15, verbose=1, validation_data=([xvalid_postText_pad, xvalid_targetDescription_pad], yvalid))


# In[ ]:


siamese_net.save_weights("/data/click_bait_detect/text_classification_model_siamese_net_attention_bidirectionlstm_epoch15.h5")


# In[1]:


#train_preds_class = siamese_net.predict_classes([xtrain_postText_pad, xtrain_targetDescription_pad])
#train_preds_class = siamese_net.predict_classes([xvalid_postText_pad, xvalid_targetDescription_pad])


# In[ ]:


train_preds = siamese_net.predict([xtrain_postText_pad, xtrain_targetDescription_pad])
valid_preds = siamese_net.predict([xvalid_postText_pad, xvalid_targetDescription_pad])


# In[50]:


fig, ax = plt.subplots(figsize=(8,6))
#print('\nTraining Accuracy:{}'.format(accuracy_score(ytrain,train_preds_class)))
#print('Mean Squared Error (MSE):{}'.format(mean_squared_error(ytrain,train_preds)))
#print('Training Confusion Matrix \n{}'.format(confusion_matrix(ytrain,train_preds_class)))
#print('Classification Report: \n{}'.format(classification_report(ytrain,train_preds_class)))

fpr,tpr,threshold=roc_curve(ytrain,train_preds)
ax.plot(fpr,tpr,label='Training AUC')
print('\nAUC:{}\n'.format(auc(fpr,tpr)))

#print ('\nTest Accuracy:{}'.format(accuracy_score(yvalid,valid_preds_class)))
#print('Mean Squared Error (MSE):{}'.format(mean_squared_error(yvalid,valid_preds)))
#print ('Test Confusion Matrix \n{}'.format(confusion_matrix(yvalid,valid_preds_class)))
#print('Classification Report: \n{}'.format(classification_report(yvalid,valid_preds_class)))
fpr,tpr,threshold=roc_curve(yvalid,valid_preds)
print ('\nAUC:{}\n'.format(auc(fpr,tpr)))
ax.plot(fpr,tpr,label='Test AUC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
ax.legend(loc='best')
plt.show()

