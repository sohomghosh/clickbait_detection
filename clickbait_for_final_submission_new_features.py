
# coding: utf-8

# In[1]:


import pandas as pd


# In[23]:


data = pd.read_csv('/data/click_bait_detect/all_features_label_exceptImages_v2_with_pos_polarity.csv')


# In[24]:


list(data.columns)


# In[2]:


raw_data_cleaned = pd.read_csv('/data/click_bait_detect/data_with_cleaned_splitted_text_new.csv')
raw_data_cleaned.head()


# In[46]:


raw_data_cleaned[(raw_data_cleaned['number_of_targetCaptions']==3) & (raw_data_cleaned['number_of_targetParagraphs']==43)]


# In[40]:


raw_data_cleaned[(raw_data_cleaned['number_of_targetParagraphs']==18) & (raw_data_cleaned['number_of_targetCaptions']==1)]


# In[44]:


raw_data_cleaned.iloc[11283,:]#8402473119094743043


# In[27]:


raw_data_cleaned[raw_data_cleaned['truthClass']=='no-clickbait'][['truthClass','number_of_targetCaptions', 'number_of_targetParagraphs', 'postText', 'targetTitle', 'targetDescription', 'targetParagraphs', 'targetKeywords', 'targetCaptions']]


# In[45]:


for ans in ['truthClass', 'number_of_targetCaptions', 'number_of_targetParagraphs', 'postText', 'targetTitle', 'targetDescription', 'targetParagraphs', 'targetKeywords', 'targetCaptions']:
    print(raw_data_cleaned.iloc[11283,:][ans], "\n")


# In[28]:


raw_data_cleaned[raw_data_cleaned['truthClass']=='clickbait'][['number_of_targetCaptions', 'number_of_targetParagraphs', 'postText', 'targetTitle', 'targetDescription', 'targetParagraphs', 'targetKeywords', 'targetCaptions']].iloc[6,:]['postText']


# In[19]:


list(raw_data_cleaned[raw_data_cleaned['truthClass']=='clickbait'][['number_of_targetCaptions', 'number_of_targetParagraphs', 'postText', 'targetTitle', 'targetDescription', 'targetParagraphs', 'targetKeywords', 'targetCaptions']].tail(1)['targetTitle'])


# In[12]:


list(raw_data_cleaned[raw_data_cleaned['truthClass']=='clickbait'][['number_of_targetCaptions', 'number_of_targetParagraphs', 'postText', 'targetTitle', 'targetDescription', 'targetParagraphs', 'targetKeywords', 'targetCaptions']].head(1)['targetDescription'])


# In[13]:


list(raw_data_cleaned[raw_data_cleaned['truthClass']=='clickbait'][['number_of_targetCaptions', 'number_of_targetParagraphs', 'postText', 'targetTitle', 'targetDescription', 'targetParagraphs', 'targetKeywords', 'targetCaptions']].head(1)['targetParagraphs'])


# In[14]:


list(raw_data_cleaned[raw_data_cleaned['truthClass']=='clickbait'][['number_of_targetCaptions', 'number_of_targetParagraphs', 'postText', 'targetTitle', 'targetDescription', 'targetParagraphs', 'targetKeywords', 'targetCaptions']].head(1)['targetKeywords'])


# In[15]:


list(raw_data_cleaned[raw_data_cleaned['truthClass']=='clickbait'][['number_of_targetCaptions', 'number_of_targetParagraphs', 'postText', 'targetTitle', 'targetDescription', 'targetParagraphs', 'targetKeywords', 'targetCaptions']].head(1)['targetCaptions'])


# In[27]:


get_ipython().system('python3 -m pip install pyemd')


# In[3]:


import gensim
import pyemd
from pyemd import emd


# In[4]:


model = gensim.models.KeyedVectors.load_word2vec_format('/data/GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[6]:


sn1 = 'Hi how are you?'
sn2 = 'How are u doing?'
distance = model.wmdistance(sn1, sn2)
print(distance)


# In[7]:


raw_data_cleaned.columns


# In[ ]:


for cm1,cm2 in [('postText_cleaned_splitted','targetTitle_cleaned_splitted'), ('postText_cleaned_splitted', 'targetDescription_cleaned_splitted'), ('postText_cleaned_splitted', 'targetParagraphs_cleaned_splitted'), ('postText_cleaned_splitted', 'targetKeywords_cleaned_splitted'), ('postText_cleaned_splitted', 'targetCaptions_cleaned_splitted'), ('targetTitle_cleaned_splitted', 'targetDescription_cleaned_splitted'), ('targetTitle_cleaned_splitted', 'targetParagraphs_cleaned_splitted'), ('targetTitle_cleaned_splitted', 'targetKeywords_cleaned_splitted'), ('targetTitle_cleaned_splitted', 'targetCaptions_cleaned_splitted')]:
    sns1 = raw_data_cleaned[cm1].apply(lambda x : ' '.join([i for i in eval(str(x))]))
    sns2 = raw_data_cleaned[cm2].apply(lambda x : ' '.join([i for i in eval(str(x))]))
    li = []
    for sn1,sn2 in zip(sns1,sns2):
        li.append(model.wmdistance(sn1, sn2))
    data['wmd_'+cm1+"_"+cm2] = li


# In[27]:


data.head()


# In[28]:


data.to_csv('/data/click_bait_detect/all_features_label_exceptImages_v2_with_pos_polarity_wmd.csv', index = False)


# ## Feature : Presence of wh words

# In[55]:


wh_words = set(['which','what','whose','who','whom','where','whither','whence','when','how','why','whether','whatsoever'])
data['postText_has_wh_words'] = raw_data_cleaned['postText_cleaned_splitted'].apply(lambda x : len(set(eval(str(x))).intersection(wh_words)))


# ## Feature :PostText has digits

# In[53]:


data['postText_has_digits'] = raw_data_cleaned['postText'].apply(lambda x : any([i.isdigit() for i in str(x)]) )


# ## Feature : Superlative Degree

# In[ ]:


#Already done while pos tags were counted

#pos_tag
'JJS':adjective, superlative
'RBS':adverb, superlative


# ## Feature : Jaccard 
# i) postText & targetTitle
# ii) postText & targetDescription

# In[ ]:


def jaccard(list_a_str, list_b_str):
    list_a = eval(str(list_a_str))
    list_b = eval(str(list_b_str))
    intersect = set(list_a).intersection(set(list_b))
    uni = set(list_a).union(set(list_b))
    try:
        jacc = len(intersect)/len(uni)
    except:
        jacc = 0
    return jacc


# In[69]:




data['postText_targetTitle_jaccard'] = raw_data_cleaned.apply(lambda row: jaccard(row['postText_cleaned_splitted'], row['targetTitle_cleaned_splitted']), axis=1)
data['postText_targetDescription_jaccard'] = raw_data_cleaned.apply(lambda row: jaccard(row['postText_cleaned_splitted'], row['targetDescription_cleaned_splitted']), axis=1)


# ## Presence of click baity phrases

# In[57]:


data['presence_of_click_baity_phrases'] = raw_data_cleaned['postText'].apply(lambda x : 1 if "click here" in str(x) or "exclusive" in str(x) or "won’t believe" in str(x) or "happens next" in str(x) or "you know" in str(x) or "don’t want" in str(x) else 0)


# In[71]:


data.to_csv('/data/click_bait_detect/all_features_label_exceptImages_v3_with_pos_polarity_wmd_extrafeatures.csv', index = False)


# # All Features

# In[2]:


list(pd.read_csv('/data/click_bait_detect/all_features_label_exceptImages_v3_with_pos_polarity_wmd_extrafeatures.csv').columns)

