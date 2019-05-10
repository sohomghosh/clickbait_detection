
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


import gc


# In[1]:


from flair.data import Sentence
from flair.models import SequenceTagger

# make a sentence
sentence = Sentence('I love Berlin .')

# load the NER tagger
tagger = SequenceTagger.load('ner')

# run NER over sentence
tagger.predict(sentence)


# In[2]:


tagger.predict(sentence)


# In[3]:


from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = '/data/click_bait_detect/glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)


# In[5]:


data = pd.read_csv('/data/click_bait_detect/data_with_cleaned_splitted_text_new.csv')
data.head()


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


fp = open('/data/click_bait_detect/targetParagraphs_cleaned_splitted_glove_vectors', 'w')
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

