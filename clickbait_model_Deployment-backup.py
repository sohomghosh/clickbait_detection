
# coding: utf-8

# In[28]:


import numpy as np
import re
from textblob import Word
from gensim.models import KeyedVectors
from scipy import spatial
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import pickle
from textblob import TextBlob
from nltk import load
import pandas as pd
from nltk.corpus import stopwords
import string
import gensim
import pyemd
from pyemd import emd
import time


# In[2]:


#Instead of writing file, compute mdedian
filename = '/data/click_bait_detect/glove.6B.50d.txt.word2vec'
model_gv = KeyedVectors.load_word2vec_format(filename, binary=False)
def get_vector(rw):
    vec_of_vec = []
    if len(rw) > 0:
        for wd in rw:
            if wd in list(model_gv.vocab.keys()):
                vec_of_vec.append(list(model_gv[wd]))
            else:
                pass
    else:
        pass
    return np.nanmean(vec_of_vec, axis = 0)#vec_of_vec


# In[3]:


def get_similarity(rw1,rw2):
    result = 1 - spatial.distance.cosine(rw1, rw2)
    return result


# In[4]:


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


# In[5]:


tagdict = load('help/tagsets/upenn_tagset.pickle')


# In[6]:


model = gensim.models.KeyedVectors.load_word2vec_format('/data/GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[7]:


lr_clf_loaded = pickle.load(open("/data/click_bait_detect/lr_clf_wmd_new_features.pickle.dat", "rb"))


# # Input

# In[8]:


number_of_postText = 1 #It's value is always 1

print("Enter number of pictures/captions in the target (i.e. news article)")
number_of_targetCaptions = 2

print("Enter number of paragraphs in the target (i.e. news article)")
number_of_targetParagraphs = 5

print("Please enter post i.e tweet")
postText = '["Apple\'s iOS 9 \'App thinning\' feature will give your phone\'s storage a boost"]'

print("Please enter title of the news artcile")
targetTitle = "Apple gives back gigabytes: iOS 9 'app thinning' feature will finally give your phone's storage a boost"

print("Please enter description of the news article")
targetDescription = "'App thinning' will be supported on Apple's iOS 9 and later models. It ensures apps use the lowest amount of storage space by 'slicing' it to work on individual handsets (illustrated)."

print("Please enter paragraphs of the news article")
targetParagraphs = '["Paying for a 64GB phone only to discover that this is significantly reduced by system files and bloatware is the bane of many smartphone owner\'s lives.??", \'And the issue became so serious earlier this year that some Apple users even sued the company over it.??\', "But with the launch of iOS 9, Apple is hoping to address storage concerns by introducing a feature known as \'app thinning.\'", \'It has been explained on the watchOS Developer Library site and is aimed at developers looking to optimise their apps to work on iOS and the watchOS.??\', \'It ensures apps use the lowest amount of storage space on a device by only downloading the parts it needs run on the particular handset it is being installed onto.\', "It \'slices\' the app into \'app variants\' that only need to access the specific files on that specific handset.??", "XperiaBlog recently spotted that the 8GB version of Sony\'s mid-range M4 Aqua has just 1.26GB of space for users.??", \'This means that firmware, pre-installed apps and Android software take up a staggering 84.25 per cent.??\', "Sony does let users increase storage space using a microSD card, but as XperiaBlog explained: \'Sony should never have launched an 8GB version of the Xperia M4 Aqua.??", "\'If you are thinking about purchasing this model, be aware of what you are buying into.\'", "Previously, apps would need to be able to run on all handsets and account for the varying files, chipsets and power so contained sections that weren\'t always relevant to the phone it was being installed on.", \'This made them larger than they needed to be.??\', \'Under the new plans, when a phone is downloaded from the App Store, the app recognises which phone it is being installed onto and only pulls in the files and code it needs to work on that particular device.??\', \'For iOS, sliced apps are supported on the latest iTunes and on devices running iOS 9.0 and later.??\', "In all other cases, the App Store will deliver the previous \'universal apps\' to customers.", "The guidelines also discuss so-called \'on-demand resources.\'??This allows developers to omit features from an app until they are opened or requested by the user.??", \'The App Store hosts these resources on Apple servers and manages the downloads for the developer and user.??\', \'This will also increase how quickly an app downloads.??\', \'An example given by Apple is a game app that may divide resources into game levels and request the next level of resources only when the app anticipates the user has completed the previous level.\', \'Similarly, the app can request In-App Purchase resources only when the user buys a corresponding in-app purchase.\', "Apple explained the operating system will then \'purge on-demand resources when they are no longer needed and disk space is low\', removing them until they are needed again.", \'And the whole iOS 9 software has been designed to be thinner during updates, namely from 4.6GB to 1.3GB, to free up space.??\', \'This app thinning applies to third-party apps created by developers.??\', "Apple doesn\'t say if it will apply to the apps Apple pre-installed on devices, such as Stocks, Weather and Safari - but it is likely that it will in order to make iOS 9 smaller.??", \'As an example of storage space on Apple devices, a 64GB Apple iPhone 6 is typically left with 56GB of free space after pre-installed apps, system files and software is included.??\', \'A drop of 8GB, leaving 87.5 per cent of storage free.??\', "By comparison, Samsung\'s 64GB S6 Edge has 53.42GB of available space, and of this 9GB is listed as system memory.??", \'Although this is a total drop of almost 11GB, it equates to 83 per cent of space free.??\', \'By comparison, on a 32GB S6 MailOnline found 23.86GB of space was available, with 6.62GB attributed to system memory.\', \'This is a drop of just over 8GB and leaves 75 per cent free.\', \'Samsung said it, too, had addressed complaints about bloatware and storage space with its S6 range. ??\', \'Previous handsets, including the Samsung Galaxy S4 and Apple iPhone 5C typically ranged from between 54 per cent and 79 per cent of free space.\', \'??\', "Businessman \'killed his best friend when he crashed jet-powered dinghy into his ??1million yacht while showing off\' as his wife filmed them"]'

print("Please enter keywords/tags relating to the news artices")
targetKeywords ='Apple,gives,gigabytes,iOS,9,app,thinning,feature,finally,phone,s,storage,boost'

print("Please enter Captions of images present in the news article")
targetCaptions = '["\'App thinning\' will be supported on Apple\'s iOS 9 and later models. It ensures apps use the lowest amount of storage space on a device by only downloading the parts it needs to run on individual handsets. It \'slices\' the app into \'app variants\' that only need to access the specific files on that specific device", "\'App thinning\' will be supported on Apple\'s iOS 9 and later models. It ensures apps use the lowest amount of storage space on a device by only downloading the parts it needs to run on individual handsets. It \'slices\' the app into \'app variants\' that only need to access the specific files on that specific device", "The guidelines also discuss so-called \'on-demand resources.\' This allows developers to omit features from an app until they are opened or requested by the user. The App Store hosts these resources on Apple servers and manages the downloads for the developer and user.??This will also increase how quickly an app downloads", "The guidelines also discuss so-called \'on-demand resources.\' This allows developers to omit features from an app until they are opened or requested by the user. The App Store hosts these resources on Apple servers and manages the downloads for the developer and user.??This will also increase how quickly an app downloads", "Apple said it will then \'purge on-demand resources when they are no longer needed and disk space is low\' (Apple\'s storage menu is pictured)", "Apple said it will then \'purge on-demand resources when they are no longer needed and disk space is low\' (Apple\'s storage menu is pictured)", \'A 64GB Apple iPhone 6 is typically left with 56GB of free space after pre-installed apps, system files and software is included. A drop of 8GB, leaving 87.5 % of storage free.??Previous handsets, including the Samsung Galaxy S4 and Apple iPhone 5C typically ranged from between 54% and 79% of free space (illustrated)\', \'A 64GB Apple iPhone 6 is typically left with 56GB of free space after pre-installed apps, system files and software is included. A drop of 8GB, leaving 87.5 % of storage free.??Previous handsets, including the Samsung Galaxy S4 and Apple iPhone 5C typically ranged from between 54% and 79% of free space (illustrated)\', "Earlier this year, a pair of disgruntled Apple users filed a lawsuit in Miami accusing the tech giant of \'concealing, omitting and failing to disclose\' that on 16GB versions of iPhones, more than 20% of the advertised space isn\'t available. This graph reveals the capacity available and unavailable to the user", "Earlier this year, a pair of disgruntled Apple users filed a lawsuit in Miami accusing the tech giant of \'concealing, omitting and failing to disclose\' that on 16GB versions of iPhones, more than 20% of the advertised space isn\'t available. This graph reveals the capacity available and unavailable to the user"]'


# In[57]:


tstart = time.time()


# In[58]:


#Vector dimension extraction
postText_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(postText)))]
targetTitle_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetTitle)))]
targetDescription_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetDescription)))]
targetParagraphs_cleaned_splitted = [Word(j).lemmatize().strip().lower() for i in eval(str(targetParagraphs)) for j in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetParagraphs)))]
targetKeywords_cleaned_splitted = str(targetKeywords).lower().split(',')
targetCaptions_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetCaptions)))]


# In[59]:


#Takes maximum time
postText_vec = get_vector(postText_cleaned_splitted)
targetTitle_vec = get_vector(targetTitle_cleaned_splitted)
targetDescription_vec = get_vector(targetDescription_cleaned_splitted)
targetParagraphs_vec = get_vector(targetParagraphs_cleaned_splitted)
targetKeywords_vec = get_vector(targetKeywords_cleaned_splitted)
targetCaptions_vec = get_vector(targetCaptions_cleaned_splitted)


# In[75]:


postText_polarity = TextBlob(str(postText)).sentiment.polarity
postText_subjectivity = TextBlob(str(postText)).sentiment.subjectivity
targetCaptions_polarity = TextBlob(str(targetCaptions)).sentiment.polarity
targetCaptions_subjectivity = TextBlob(str(targetCaptions)).sentiment.subjectivity
targetDescription_polarity = TextBlob(str(targetDescription)).sentiment.polarity
targetDescription_subjectivity = TextBlob(str(targetDescription)).sentiment.subjectivity
targetParagraphs_polarity = TextBlob(str(targetParagraphs)).sentiment.polarity
targetParagraphs_subjectivity = TextBlob(str(targetParagraphs)).sentiment.subjectivity
targetTitle_polarity = TextBlob(str(targetTitle)).sentiment.polarity
targetTitle_subjectivity = TextBlob(str(targetTitle)).sentiment.subjectivity


# In[76]:


pos_di = {}
for pos in list(tagdict.keys()):
	pos_di[pos] = []

di = Counter([j for i,j in pos_tag(word_tokenize(postText))])
for pos in list(tagdict.keys()):
	pos_di[pos].append(di[pos])

postTitle_pos_df = pd.DataFrame(pos_di)

#number of stop words
stp_wds = set(stopwords.words('english'))
postText_number_of_stop_words = len(stp_wds.intersection(word_tokenize(str(postText))))

#number of punctations
postText_num_of_unique_punctuations = len(set(postText).intersection(set(string.punctuation)))


# In[77]:


#similarity : SEE ABOVE
sim_posttext_targetTitle = get_similarity(postText_vec, targetTitle_vec)
sim_posttext_targetDescription = get_similarity(postText_vec,targetDescription_vec)
sim_posttext_targetParagraphs = get_similarity(postText_vec,targetParagraphs_vec)
sim_posttext_targetKeywords = get_similarity(postText_vec,targetKeywords_vec)
sim_posttext_targetCaptions = get_similarity(postText_vec,targetCaptions_vec)
sim_targetTitle_targetDescription = get_similarity(targetTitle_vec, targetDescription_vec)
sim_targetTitle_targetParagraphs = get_similarity(targetTitle_vec, targetParagraphs_vec)
sim_targetTitle_targetKeywords = get_similarity(targetTitle_vec, targetKeywords_vec)
sim_targetTitle_targetCaptions = get_similarity(targetTitle_vec, targetCaptions_vec)


# In[78]:


test_df = pd.DataFrame({'number_of_postText' : [number_of_postText], 'number_of_targetCaptions' : [number_of_targetCaptions], 'number_of_targetParagraphs' : [number_of_targetParagraphs],
             'sim_posttext_targetDescription' : [sim_posttext_targetDescription],
    'sim_posttext_targetParagraphs' : [sim_posttext_targetParagraphs],
 'sim_posttext_targetKeywords' : [sim_posttext_targetKeywords],
 'sim_posttext_targetCaptions' : [sim_posttext_targetCaptions],
 'sim_targetTitle_targetDescription' : [sim_targetTitle_targetDescription],
 'sim_targetTitle_targetParagraphs' : [sim_targetTitle_targetParagraphs],
 'sim_targetTitle_targetKeywords' : [sim_targetTitle_targetKeywords],
 'sim_targetTitle_targetCaptions' : [sim_targetTitle_targetCaptions],
 'postText_polarity' : [postText_polarity],
 'postText_subjectivity' : [postText_subjectivity],
 'targetCaptions_polarity' : [targetCaptions_polarity],
 'targetCaptions_subjectivity' : [targetCaptions_subjectivity],
 'targetDescription_polarity' : [targetDescription_polarity],
 'targetDescription_subjectivity' : [targetDescription_subjectivity],
 'targetParagraphs_polarity' : [targetParagraphs_polarity],
 'targetParagraphs_subjectivity' : [targetParagraphs_subjectivity],
 'targetTitle_polarity' : [targetTitle_polarity],
 'targetTitle_subjectivity' : [targetTitle_subjectivity],
 'postText_num_of_unique_punctuations' : [postText_num_of_unique_punctuations],
 'postText_number_of_stop_words' : [postText_number_of_stop_words],
 'sim_posttext_targetTitle' : [sim_posttext_targetTitle]
             })
test_df = pd.concat([test_df,postTitle_pos_df], axis = 1)


# In[79]:


for vec in ['postText_vec', 'targetTitle_vec', 'targetDescription_vec', 'targetParagraphs_vec', 'targetKeywords_vec', 'targetCaptions_vec']:
    for itm,dim in zip(eval(vec),range(50)):
        test_df[vec[:-4]+'_golve_dim'+str(dim)] = [itm]


# In[80]:


for cm1,cm2 in [('postText_cleaned_splitted','targetTitle_cleaned_splitted'), ('postText_cleaned_splitted', 'targetDescription_cleaned_splitted'), ('postText_cleaned_splitted', 'targetParagraphs_cleaned_splitted'), ('postText_cleaned_splitted', 'targetKeywords_cleaned_splitted'), ('postText_cleaned_splitted', 'targetCaptions_cleaned_splitted'), ('targetTitle_cleaned_splitted', 'targetDescription_cleaned_splitted'), ('targetTitle_cleaned_splitted', 'targetParagraphs_cleaned_splitted'), ('targetTitle_cleaned_splitted', 'targetKeywords_cleaned_splitted'), ('targetTitle_cleaned_splitted', 'targetCaptions_cleaned_splitted')]:
    sns1 = ' '.join([i for i in eval(cm1)])
    sns2 = ' '.join([i for i in eval(cm2)])
    dist = model.wmdistance(sns1, sns2)
    new_df = pd.DataFrame({'wmd_'+cm1+"_"+cm2 : [dist]})
    test_df = pd.concat([test_df,new_df], axis =1)


# In[81]:


wh_words = set(['which','what','whose','who','whom','where','whither','whence','when','how','why','whether','whatsoever'])
postText_has_wh_words = len(set(postText_cleaned_splitted).intersection(wh_words))


# In[82]:


postText_has_digits = any([i.isdigit() for i in postText])


# In[83]:


postText_targetTitle_jaccard = jaccard(postText_cleaned_splitted, targetTitle_cleaned_splitted)
postText_targetDescription_jaccard = jaccard(postText_cleaned_splitted, targetDescription_cleaned_splitted)


# In[84]:


x = postText
presence_of_click_baity_phrases = 1 if "click here" in str(x) or "exclusive" in str(x) or "won’t believe" in str(x) or "happens next" in str(x) or "you know" in str(x) or "don’t want" in str(x) else 0


# In[85]:


new_features_df = pd.DataFrame({'postText_has_wh_words':[postText_has_wh_words], 'postText_has_digits':[postText_has_digits], 'postText_targetTitle_jaccard':[postText_targetTitle_jaccard], 'postText_targetDescription_jaccard':[postText_targetDescription_jaccard], 'presence_of_click_baity_phrases':[presence_of_click_baity_phrases]})
test_df = pd.concat([test_df, new_features_df], axis = 1)
test_df = test_df.drop(['$',')', 'EX', 'FW', 'JJR', 'JJS', 'NNPS', 'RBR', 'RBS', 'RP', 'UH', 'WDT', 'WP', 'WRB', 'sim_posttext_targetKeywords', 'sim_targetTitle_targetCaptions', 'sim_targetTitle_targetKeywords', 'targetKeywords_golve_dim16', 'targetKeywords_golve_dim26', 'targetKeywords_golve_dim32', 'targetKeywords_golve_dim33', 'targetKeywords_golve_dim35','wmd_postText_cleaned_splitted_targetKeywords_cleaned_splitted', 'wmd_targetTitle_cleaned_splitted_targetKeywords_cleaned_splitted'], axis = 1)


# In[86]:


lr_clf_loaded.predict_proba(test_df)[:,1]


# In[87]:


tend = time.time()


# In[88]:


print((tend - tstart ), 'seconds')


# In[89]:


441/60

