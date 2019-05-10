
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[30]:


from nltk import word_tokenize


# In[31]:


from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer


# In[32]:


import pickle


# In[34]:


import re


# In[15]:


import seaborn as sns
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# # File parsing

# In[ ]:


data location
data(/click_bait_detect/clickbait17-train-170331)
truth.jsonl
instances.jsonl


# In[18]:


pd.read_json('/data/click_bait_detect/clickbait17-train-170331/instances.jsonl', lines=True).to_csv('/data/click_bait_detect/clickbait17-train-170331/instances.csv', index = False)


# In[22]:


pd.read_json('/data/click_bait_detect/clickbait17-train-170331/instances.jsonl', lines=True).shape


# In[23]:


pd.read_csv('/data/click_bait_detect/clickbait17-train-170331/instances.csv').shape


# In[21]:


pd.read_json('/data/click_bait_detect/clickbait17-train-170331/truth.jsonl', lines=True).to_csv('/data/click_bait_detect/clickbait17-train-170331/truth.csv', index = False)


# In[24]:


pd.read_json('/data/click_bait_detect/clickbait17-train-170331/truth.jsonl', lines=True).shape


# In[25]:


pd.read_csv('/data/click_bait_detect/clickbait17-train-170331/truth.csv').shape


# In[15]:


i = 0
for line in open("/data/click_bait_detect/clickbait17-train-170331/instances.jsonl", 'rb'):
    for itm in ['id', 'postTimestamp', 'postText', 'postMedia', 'targetTitle', 'targetDescription', 'targetKeywords', 'targetParagraphs', 'targetCaptions']:
        print(eval(line)[itm])
        i = i+1
    if i ==2:
        break


# In[ ]:


id,postTimestamp,postText,postMedia,targetTitle,targetDescription,targetKeywords,targetParagraphs,targetCaptions



{ 
    "id": "<instance id>",
    "postTimestamp": "<weekday> <month> <day> <hour>:<minute>:<second> <time_offset> <year>",
    "postText": ["<text of the post with links removed>"],
    "postMedia": ["<path to a file in the media archive>"],
    "targetTitle": "<title of target article>",
    "targetDescription": "<description tag of target article>",
    "targetKeywords": "<keywords tag of target article>",
    "targetParagraphs": ["<text of the ith paragraph in the target article>"],
    "targetCaptions": ["<caption of the ith image in the target article>"]
  }


# In[27]:


pd.read_json('/data/click_bait_detect/clickbait17-validation-170630/instances.jsonl', lines=True).to_csv('/data/click_bait_detect/clickbait17-validation-170630/instances_validation.csv', index = False)


# In[29]:


pd.read_json('/data/click_bait_detect/clickbait17-validation-170630/truth.jsonl', lines=True).to_csv('/data/click_bait_detect/clickbait17-validation-170630/truth_validation.csv', index = False)


# In[30]:


validation_truth = pd.read_csv('/data/click_bait_detect/clickbait17-validation-170630/truth_validation.csv')
validation_instances = pd.read_csv('/data/click_bait_detect/clickbait17-validation-170630/instances_validation.csv')
train_truth = pd.read_csv('/data/click_bait_detect/clickbait17-train-170331/truth.csv')
train_instances = pd.read_csv('/data/click_bait_detect/clickbait17-train-170331/instances.csv')


# In[36]:


train_instances.columns


# In[39]:


train_instances.merge(train_truth, on = 'id', how = 'inner').to_csv('/data/click_bait_detect/clickbait17-train-170331/train_truth_instances.csv', index = False)


# In[42]:


validation_instances.merge(validation_truth, on = 'id', how = 'inner').to_csv('/data/click_bait_detect/clickbait17-validation-170630/validation_truth_instances.csv', index = False)


# In[3]:


train = pd.read_csv('/data/click_bait_detect/clickbait17-train-170331/train_truth_instances.csv')
validation = pd.read_csv('/data/click_bait_detect/clickbait17-validation-170630/validation_truth_instances.csv')


# In[4]:


train.head()


# In[8]:


np.mean([0.0, 0.6666667, 0.0, 0.33333334000000003, 0.0])


# In[6]:


train.append(validation).to_csv('/data/click_bait_detect/train_validation_truth_instances.csv', index = False)


# In[4]:


data = pd.read_csv('/data/click_bait_detect/train_validation_truth_instances.csv')


# In[5]:


data.head()


# In[11]:


data['truthClass'].value_counts(normalize = True)


# In[26]:


sns.boxplot(data=data[data['truthClass']=='clickbait'], x = 'truthMean')
plt.xlim(0,1)
plt.title('For click bait posts')
plt.show()


# In[30]:


sns.boxplot(data=data[data['truthClass']=='no-clickbait'], x = 'truthMean')
plt.xlim(0,1)
plt.title('For not click bait posts')
plt.show()


# In[25]:


sns.boxplot(data=data[data['truthClass']=='clickbait'], x = 'truthMedian')
plt.xlim(0,1)
plt.title('For click bait posts')
plt.show()


# In[29]:


sns.boxplot(data=data[data['truthClass']=='no-clickbait'], x = 'truthMedian')
plt.xlim(0,1)
plt.title('For not click bait posts')
plt.show()


# In[27]:


sns.boxplot(data=data[data['truthClass']=='clickbait'], x = 'truthMode')
plt.xlim(0,1)
plt.title('For click bait posts')
plt.show()


# In[28]:


sns.boxplot(data=data[data['truthClass']=='no-clickbait'], x = 'truthMedian')
plt.xlim(0,1)
plt.title('For not click bait posts')
plt.show()


# In[13]:


data['postMedia'].apply(lambda x: len(eval(x))).value_counts()


# In[15]:


data['number_of_images'] = data['postMedia'].apply(lambda x: len(eval(x)))


# In[22]:


data.groupby(['number_of_images', 'truthClass']).size().unstack()


# In[23]:


data['number_of_postText'] = data['postText'].apply(lambda x: len(eval(x)))
data.groupby(['number_of_postText', 'truthClass']).size().unstack()


# In[43]:


data['number_of_targetCaptions'] = data['targetCaptions'].apply(lambda x: len(eval(x)))
data.groupby(['number_of_targetCaptions', 'truthClass']).size().unstack().head()


# In[31]:


data.groupby(['number_of_targetCaptions', 'truthClass']).size().unstack().to_csv('numCaptions_truthClass.csv')


# In[40]:


data['number_of_targetKeyWords'] = data['targetKeywords'].apply(lambda x : 0 if str(x)=='nan' else len(x.split(',')))


# In[42]:


data.groupby(['number_of_targetKeyWords', 'truthClass']).size().unstack().to_csv('numKeywords_truthClass.csv')


# In[46]:


data['number_of_targetParagraphs'] = data['targetParagraphs'].apply(lambda x : 0 if str(x)=='nan' else len(x.split(',')))
data.groupby(['number_of_targetParagraphs', 'truthClass']).size().unstack().to_csv('numtargetParagraphs_truthClass.csv')


# # Feature extraction from time stamp

# In[47]:


data['DateTime'] =  pd.to_datetime(data['postTimestamp'], errors='coerce')
data['weekday'] = data['DateTime'].dt.weekday
data['day_of_month'] = data['DateTime'].dt.day
data['month'] = data['DateTime'].dt.month
data['hour'] = data['DateTime'].dt.hour
data['year'] = data['DateTime'].dt.year


# In[56]:


data.groupby(['day_of_month', 'truthClass']).size().unstack()


# In[57]:


data.groupby(['hour', 'truthClass']).size().unstack()


# In[53]:


data.groupby(['weekday', 'truthClass']).size().unstack()


# In[54]:


data.groupby(['month', 'truthClass']).size().unstack()


# In[55]:


data.groupby(['year', 'truthClass']).size().unstack()


# # Feature extraction from texts
# ## text cleaning
# 
# ## tfidf
# ## w2v
# 
# ## GloVe word embeddings
# 
# ## Bi-directional LSTM to find similarity between postText and targetTitle

# In[58]:


data.columns


# In[59]:


data['postText'].head()


# In[37]:


import nltk
nltk.download('punkt')


# In[36]:


from textblob import Word


# In[89]:


import re


# In[109]:


[Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', list(data['postText'])[409]))]


# In[ ]:


#Text columns

##single sentence
postText
targetTitle
targetDescription

##List of words/senetnces
targetKeywords
targetParagraphs : ["<text of the ith paragraph in the target article>"]

##Other Texts
targetCaptions : ["<caption of the ith image in the target article>"]


# In[114]:


data['targetTitle_cleaned_splitted'] = data['targetTitle'].apply(lambda x : [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(x)))])


# In[117]:


data['postText_cleaned_splitted'] = data['postText'].apply(lambda x : [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(x)))])


# In[125]:


data['targetDescription_cleaned_splitted'] = data['targetDescription'].apply(lambda x : [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(x)))])


# In[138]:


data['targetKeywords_cleaned_splitted'] = data['targetKeywords'].apply(lambda x : str(x).lower().split(','))


# In[154]:


data['targetParagraphs_cleaned_splitted'] = data['targetParagraphs'].apply(lambda x : [Word(j).lemmatize().strip().lower() for i in eval(str(x)) for j in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(i)))])


# In[155]:


data.to_csv('/data/click_bait_detect/data_with_cleaned_splitted_text.csv', index = False)


# In[9]:


data = pd.read_csv('/data/click_bait_detect/data_with_cleaned_splitted_text_new.csv')
data.head()


# In[10]:


data['weekday']


# ### Using Glove

# In[48]:


from gensim.scripts.glove2word2vec import glove2word2vec


# In[49]:


glove_input_file = '/data/click_bait_detect/glove.6B.50d.txt'
word2vec_output_file = '/data/click_bait_detect/glove.6B.50d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)


# In[50]:


from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = '/data/click_bait_detect/glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)


# In[230]:


model['data']


# In[203]:


import gc


# In[ ]:


fp = open('/data/click_bait_detect/targetTitle_cleaned_splitted_glove_vectors', 'w')
for rw in data['targetTitle_cleaned_splitted']:
    vec_of_vec = []
    if len(rw) > 0:
        for wd in rw:
            if wd in list(model.vocab.keys()):
                vec_of_vec.append(list(model[wd]))
            else:
                pass
    else:
        pass
    fp.write(str(vec_of_vec)+'\n')
    del vec_of_vec
    gc.collect()
fp.close()


# In[ ]:


fp = open('/data/click_bait_detect/postText_cleaned_splitted_glove_vectors', 'w')
for rw in data['postText_cleaned_splitted']:
    vec_of_vec = []
    if len(rw) > 0:
        for wd in rw:
            if wd in list(model.vocab.keys()):
                vec_of_vec.append(list(model[wd]))
            else:
                pass
    else:
        pass
    fp.write(str(vec_of_vec)+'\n')
    del vec_of_vec
    gc.collect()
fp.close()


# In[ ]:


fp = open('/data/click_bait_detect/targetDescription_cleaned_splitted_glove_vectors', 'w')
for rw in data['targetDescription_cleaned_splitted']:
    vec_of_vec = []
    if len(rw) > 0:
        for wd in rw:
            if wd in list(model.vocab.keys()):
                vec_of_vec.append(list(model[wd]))
            else:
                pass
    else:
        pass
    fp.write(str(vec_of_vec)+'\n')
    del vec_of_vec
    gc.collect()
fp.close()


# In[ ]:


fp = open('/data/click_bait_detect/targetKeywords_cleaned_splitted_glove_vectors', 'w')
for rw in data['targetKeywords_cleaned_splitted']:
    vec_of_vec = []
    if len(rw) > 0:
        for wd in rw:
            if wd in list(model.vocab.keys()):
                vec_of_vec.append(list(model[wd]))
            else:
                pass
    else:
        pass
    fp.write(str(vec_of_vec)+'\n')
    del vec_of_vec
    gc.collect()
fp.close()


# In[ ]:


filename = '/data/click_bait_detect/glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
fp = open('/data/click_bait_detect/targetParagraphs_cleaned_splitted_glove_vectors_new', 'w')
for rw in data['targetParagraphs_cleaned_splitted']:
    vec_of_vec = []
    if len(rw) > 0:
        for wd in rw:
            if wd in list(model.vocab.keys()):
                vec_of_vec.append(list(model[wd]))
            else:
                pass
    else:
        pass
    fp.write(str(vec_of_vec)+'\n')
    del vec_of_vec
    gc.collect()
fp.close()


# In[53]:


import gc


# In[ ]:


filename = '/data/click_bait_detect/glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
fp = open('/data/click_bait_detect/targetCaptions_cleaned_splitted_glove_vectors_new', 'w')
for rw in data['targetCaptions_cleaned_splitted']:
    vec_of_vec = []
    if len(rw) > 0:
        for wd in rw:
            if wd in list(model.vocab.keys()):
                vec_of_vec.append(list(model[wd]))
            else:
                pass
    else:
        pass
    fp.write(str(vec_of_vec)+'\n')
    del vec_of_vec
    gc.collect()
fp.close()


# In[165]:


data.columns


# ## Using Word2vec

# In[43]:


col = 'targetCaptions_cleaned_splitted'
sentences_split = list(data[col])
model_w2v = word2vec.Word2Vec(sentences_split, size=50,min_count =1, window=3, workers =-1,sample=1e-5)
features_sent = np.zeros(shape=(0,50))
for i in sentences_split:
	su=np.zeros(shape=(50))
	num_words = 0
	for j in i:
		k=np.array(model_w2v.wv[j])
		su=su+k
		#print(su)
		num_words = num_words + 1
	features_sent=np.vstack([features_sent, su/num_words])

np.savetxt('/data/click_bait_detect/features_sent_w2v_'+col+'.txt',features_sent)


# In[44]:


model_w2v.save("/data/click_bait_detect/word2vec_targetCaptions_cleaned_splitted.model")


# In[7]:


#save word2vec model for future
for col in ['targetTitle_cleaned_splitted','postText_cleaned_splitted', 'targetDescription_cleaned_splitted','targetKeywords_cleaned_splitted', 'targetParagraphs_cleaned_splitted']:
    sentences_split = list(data[col])
    model_w2v = word2vec.Word2Vec(sentences_split, size=50,min_count =1, window=3, workers =-1,sample=1e-5)
    model_w2v.save("/data/click_bait_detect/word2vec_"+col+".model")


# ## Using TF-IDF

# In[47]:


for col in ['targetCaptions_cleaned_splitted']:#['targetTitle_cleaned_splitted','postText_cleaned_splitted', 'targetDescription_cleaned_splitted','targetKeywords_cleaned_splitted', 'targetParagraphs_cleaned_splitted']:
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=.01, use_idf=True, smooth_idf=False, sublinear_tf=True)
    model_tfidf = sklearn_tfidf.fit(data[col])
    data_col_tfidf = pd.DataFrame(sklearn_tfidf.fit_transform(data[col]).todense())
    np.savetxt('/data/click_bait_detect/features_sent_tfidf_'+col+'.txt',data_col_tfidf)
    pickle.dump(model_tfidf, open("/data/click_bait_detect/tfidf_"+col+".pickle.dat", "wb"))


# In[11]:


#save/pickle tfidf model for future
for col in ['targetTitle_cleaned_splitted','postText_cleaned_splitted', 'targetDescription_cleaned_splitted','targetKeywords_cleaned_splitted', 'targetParagraphs_cleaned_splitted']:
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=.01, use_idf=True, smooth_idf=False, sublinear_tf=True)
    model_tfidf = sklearn_tfidf.fit(data[col])
    pickle.dump(model_tfidf, open("/data/click_bait_detect/tfidf_"+col+".pickle.dat", "wb"))


# # Save in
# dividual tfidf,w2v models for doing analysis of test set

# # Feature extraction from images

# In[ ]:


#Image locations
postMedia


# # Training regression/classification model : Predict probability
# ## Logistic Regression
# ## Random Forest
# ## XG-Boost
# ## Catboost
# ## LightGBM
# ## Deep Learning

# In[6]:


data.columns


# In[15]:


data_use = data[['truthClass', 'number_of_images', 'number_of_postText',
       'number_of_targetCaptions', 'number_of_targetKeyWords',
       'number_of_targetParagraphs','weekday', 'day_of_month',
       'month', 'hour', 'year']]
data_use.head()


# In[25]:


data_features_label_1 = pd.concat([pd.get_dummies(data_use[['year','month','weekday']].astype(str,errors='ignore')), data_use[['number_of_images', 'number_of_postText', 'number_of_targetCaptions', 'number_of_targetKeyWords', 'number_of_targetParagraphs', 'day_of_month', 'hour', 'truthClass']]],axis = 1)
data_features_label_1.head()


# In[26]:


data_features_label_1.to_csv('/data/click_bait_detect/data_timestamp_count_features_label.csv', index = False)


# In[ ]:


data_features_


# In[ ]:


'targetTitle_cleaned_splitted',
       'postText_cleaned_splitted', 'targetDescription_cleaned_splitted',
       'targetKeywords_cleaned_splitted', 'targetParagraphs_cleaned_splitted'


# In[ ]:


'postMedia'


# In[51]:


data['targetCaptions_cleaned_splitted'] = data['targetCaptions'].apply(lambda x : [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(x)))])


# In[41]:


data.to_csv('/data/click_bait_detect/data_with_cleaned_splitted_text_new.csv', index = False)


# In[42]:


get_ipython().system('cp /data/click_bait_detect/data_with_cleaned_splitted_text_new.csv /data/kaggle_data_ga_crp/')


# In[28]:


data.columns


# # Evaluation
# ## Mean Squared Error (MSE) with respect to the mean judgments of the annotators is used. For informational purposes, we compute further evaluation metrics such as the Median Absolute Error (MedAE), the F1-Score (F1) with respect to the truth class, as well as the runtime of the classification software
