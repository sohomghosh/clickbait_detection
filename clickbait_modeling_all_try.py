
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from datetime import datetime
from IPython import get_ipython
import seaborn as sns
from sklearn.feature_selection import RFECV,RFE,VarianceThreshold
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_curve,classification_report,confusion_matrix,precision_recall_curve,auc, mean_squared_error
from sklearn.preprocessing import StandardScaler,scale,Normalizer
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,cross_val_score
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale,StandardScaler,RobustScaler,Normalizer
from sklearn.feature_selection import SelectKBest,f_regression,f_classif,SelectFromModel
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from textblob import TextBlob
get_ipython().magic('matplotlib inline')
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from collections import Counter
from nltk.data import load
import re
from textblob import Word
import pickle


# In[122]:


data_features_label_1 = pd.read_csv('/data/click_bait_detect/data_timestamp_count_features_label.csv')
data_features_label_1.head()


# In[123]:


data_features_label_1.columns


# In[124]:


data_features_label_1 = data_features_label_1.drop(['year_2015', 'year_2016', 'year_2017', 'month_1', 'month_12', 'month_2',
       'month_3', 'month_4', 'month_6', 'weekday_0', 'weekday_1', 'weekday_2',
       'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'day_of_month', 'hour', 'number_of_images', 'number_of_targetKeyWords'], axis = 1)


# In[125]:


data_features_label_1.columns


# In[37]:


feature = 'postText'
features_sent_sub = np.zeros(shape=(0,50))
count_ok = 0
count_not_ok = 0
for line in open('/data/click_bait_detect/'+ feature +'_cleaned_splitted_glove_vectors','r'):
    if len(eval(line))>0:
        count_ok = count_ok + 1
        features_sent_sub= np.vstack([features_sent_sub, np.nanmean(eval(line), axis = 0)])
    else:
        count_not_ok = count_not_ok + 1
        features_sent_sub= np.vstack([features_sent_sub, np.zeros(shape=(1,50))])


# In[38]:


features_sent_sub_pt = features_sent_sub


# In[66]:


np.savetxt('/data/click_bait_detect/posttext_glove_vectors_average.txt',features_sent_sub_pt)


# In[5]:


features_sent_sub_pt = np.loadtxt('/data/click_bait_detect/posttext_glove_vectors_average.txt')


# In[7]:


features_sent_sub_pt.shape


# In[10]:


count_not_ok


# In[8]:


count_ok


# In[8]:


need_features = pd.DataFrame(features_sent_sub_pt)
need_features.columns = ['postText_golve_dim'+str(i) for i in need_features.columns]
need_features.head()


# In[12]:


feature = 'targetTitle'
features_sent_sub = np.zeros(shape=(0,50))
count_ok = 0
count_not_ok = 0
for line in open('/data/click_bait_detect/'+ feature +'_cleaned_splitted_glove_vectors','r'):
    if len(eval(line))>0:
        count_ok = count_ok + 1
        features_sent_sub= np.vstack([features_sent_sub, np.nanmean(eval(line), axis = 0)])
    else:
        count_not_ok = count_not_ok + 1
        features_sent_sub= np.vstack([features_sent_sub, np.zeros(shape=(1,50))])


# In[13]:


features_sent_sub_tt = features_sent_sub


# In[67]:


np.savetxt('/data/click_bait_detect/targetTitle_glove_vectors_average.txt',features_sent_sub_tt)


# In[9]:


features_sent_sub_tt = np.loadtxt('/data/click_bait_detect/targetTitle_glove_vectors_average.txt') 


# In[10]:


need_features_tt = pd.DataFrame(features_sent_sub_tt)
need_features_tt.columns = ['targetTitle_golve_dim'+str(i) for i in need_features_tt.columns]
need_features_tt.head()


# # Similarity between posttext and targetTitle

# In[11]:


from scipy import spatial
cnt = 0
sim_posttext_targetTitle = []
for rw1,rw2 in zip(features_sent_sub_pt, features_sent_sub_tt):
    result = 1 - spatial.distance.cosine(rw1, rw2)
    sim_posttext_targetTitle.append(result)


# In[12]:


da = pd.concat([data_features_label_1,need_features,need_features_tt], axis = 1)
da['sim_posttext_targetTitle'] = sim_posttext_targetTitle


# In[13]:


data_features_label_1.shape


# In[14]:


da.head()


# In[15]:


pd.isnull(da['sim_posttext_targetTitle']).value_counts()


# # Using other features: targetDescription, sim(posttext, targetDescription), targetParagraphs, sim(posttext, targetParagraphs), targetKeywords, targetCaptions, image features, sim(posttext,targetKeywords), sim(posttext,targetCaptions), sim(targetTitle,targetDescription), sim(targetTitle,targetParagraphs), sim(targetTitle,targetKeywords),sim(targetTitle,targetCaptions)

# In[30]:


feature = 'targetDescription'
features_sent_sub = np.zeros(shape=(0,50))
count_ok = 0
count_not_ok = 0
for line in open('/data/click_bait_detect/'+ feature +'_cleaned_splitted_glove_vectors','r'):
    if len(eval(line))>0:
        count_ok = count_ok + 1
        features_sent_sub= np.vstack([features_sent_sub, np.nanmean(eval(line), axis = 0)])
    else:
        count_not_ok = count_not_ok + 1
        features_sent_sub= np.vstack([features_sent_sub, np.zeros(shape=(1,50))])

features_sent_sub_td = features_sent_sub


# In[68]:


np.savetxt('/data/click_bait_detect/targetDescription_glove_vectors_average.txt',features_sent_sub_td)


# In[16]:


features_sent_sub_td = np.loadtxt('/data/click_bait_detect/targetDescription_glove_vectors_average.txt')
need_features_td = pd.DataFrame(features_sent_sub_td)
need_features_td.columns = ['targetDescription_golve_dim'+str(i) for i in need_features_td.columns]
da = pd.concat([da, need_features_td], axis = 1)


# In[33]:


feature = 'targetParagraphs'
features_sent_sub = np.zeros(shape=(0,50))
count_ok = 0
count_not_ok = 0
for line in open('/data/click_bait_detect/'+ feature +'_cleaned_splitted_glove_vectors_new','r'):
    if len(eval(line))>0:
        count_ok = count_ok + 1
        features_sent_sub= np.vstack([features_sent_sub, np.nanmean(eval(line), axis = 0)])
    else:
        count_not_ok = count_not_ok + 1
        features_sent_sub= np.vstack([features_sent_sub, np.zeros(shape=(1,50))])

features_sent_sub_tp = features_sent_sub


# In[69]:


np.savetxt('/data/click_bait_detect/targetParagraphs_glove_vectors_average.txt',features_sent_sub_tp)


# In[17]:


features_sent_sub_tp = np.loadtxt('/data/click_bait_detect/targetParagraphs_glove_vectors_average.txt')
need_features_tp = pd.DataFrame(features_sent_sub_tp)
need_features_tp.columns = ['targetParagraphs_golve_dim'+str(i) for i in need_features_tp.columns]
da = pd.concat([da, need_features_tp], axis = 1)


# In[35]:


#need_features_tp.to_csv('/data/click_bait_detect/targetParagraphs_glove_vectors_average.csv', index = False)


# In[39]:


feature = 'targetKeywords'
features_sent_sub = np.zeros(shape=(0,50))
count_ok = 0
count_not_ok = 0
for line in open('/data/click_bait_detect/'+ feature +'_cleaned_splitted_glove_vectors','r'):
    if len(eval(line))>0:
        count_ok = count_ok + 1
        features_sent_sub= np.vstack([features_sent_sub, np.nanmean(eval(line), axis = 0)])
    else:
        count_not_ok = count_not_ok + 1
        features_sent_sub= np.vstack([features_sent_sub, np.zeros(shape=(1,50))])

features_sent_sub_tk = features_sent_sub


# In[70]:


np.savetxt('/data/click_bait_detect/targetKeywords_glove_vectors_average.txt',features_sent_sub_tk)


# In[18]:


features_sent_sub_tk = np.loadtxt('/data/click_bait_detect/targetKeywords_glove_vectors_average.txt')
need_features_tk = pd.DataFrame(features_sent_sub_tk)
need_features_tk.columns = ['targetKeywords_golve_dim'+str(i) for i in need_features_tk.columns]
da = pd.concat([da, need_features_tk], axis = 1)


# In[41]:


feature = 'targetCaptions'
features_sent_sub = np.zeros(shape=(0,50))
count_ok = 0
count_not_ok = 0
for line in open('/data/click_bait_detect/'+ feature +'_cleaned_splitted_glove_vectors_new','r'):
    if len(eval(line))>0:
        count_ok = count_ok + 1
        features_sent_sub= np.vstack([features_sent_sub, np.nanmean(eval(line), axis = 0)])
    else:
        count_not_ok = count_not_ok + 1
        features_sent_sub= np.vstack([features_sent_sub, np.zeros(shape=(1,50))])

features_sent_sub_tc = features_sent_sub


# In[71]:


np.savetxt('/data/click_bait_detect/targetCaptions_glove_vectors_average.txt',features_sent_sub_tc)


# In[19]:


features_sent_sub_tc = np.loadtxt('/data/click_bait_detect/targetCaptions_glove_vectors_average.txt')
need_features_tc = pd.DataFrame(features_sent_sub_tc)
need_features_tc.columns = ['targetCaptions_golve_dim'+str(i) for i in need_features_tc.columns]
da = pd.concat([da, need_features_tc], axis = 1)


# # Calculating sim(posttext, targetDescription)

# In[20]:


from scipy import spatial
cnt = 0
sim = []
for rw1,rw2 in zip(features_sent_sub_pt, features_sent_sub_td):
    result = 1 - spatial.distance.cosine(rw1, rw2)
    sim.append(result)

da['sim_posttext_targetDescription'] = sim


# # Calculating sim(posttext, targetParagraphs)

# In[21]:


from scipy import spatial
cnt = 0
sim = []
for rw1,rw2 in zip(features_sent_sub_pt, features_sent_sub_tp):
    result = 1 - spatial.distance.cosine(rw1, rw2)
    sim.append(result)

da['sim_posttext_targetParagraphs'] = sim


# # Calculating sim(posttext,targetKeywords)

# In[22]:


cnt = 0
sim = []
for rw1,rw2 in zip(features_sent_sub_pt, features_sent_sub_tk):
    result = 1 - spatial.distance.cosine(rw1, rw2)
    sim.append(result)

da['sim_posttext_targetKeywords'] = sim


# # Calculating sim(posttext,targetCaptions)

# In[23]:


cnt = 0
sim = []
for rw1,rw2 in zip(features_sent_sub_pt, features_sent_sub_tc):
    result = 1 - spatial.distance.cosine(rw1, rw2)
    sim.append(result)

da['sim_posttext_targetCaptions'] = sim


# # Calculating sim(targetTitle,targetDescription)

# In[24]:


sim = []
for rw1,rw2 in zip(features_sent_sub_tt, features_sent_sub_td):
    result = 1 - spatial.distance.cosine(rw1, rw2)
    sim.append(result)

da['sim_targetTitle_targetDescription'] = sim


# # Calculating sim(targetTitle,targetParagraphs)

# In[25]:


sim = []
for rw1,rw2 in zip(features_sent_sub_tt, features_sent_sub_tp):
    result = 1 - spatial.distance.cosine(rw1, rw2)
    sim.append(result)

da['sim_targetTitle_targetParagraphs'] = sim


# # Calculating sim(targetTitle,targetKeywords)

# In[26]:


sim = []
for rw1,rw2 in zip(features_sent_sub_tt, features_sent_sub_tk):
    result = 1 - spatial.distance.cosine(rw1, rw2)
    sim.append(result)

da['sim_targetTitle_targetKeywords'] = sim


# # Calculating sim(targetTitle,targetCaptions)

# In[27]:


sim = []
for rw1,rw2 in zip(features_sent_sub_tt, features_sent_sub_tc):
    result = 1 - spatial.distance.cosine(rw1, rw2)
    sim.append(result)

da['sim_targetTitle_targetCaptions'] = sim


# In[28]:


da = da.fillna(0)


# In[29]:


da['truthClass_numeric'] =  da['truthClass'].replace(to_replace = {'no-clickbait': 0, 'clickbait': 1})


# In[53]:


#da.drop('truthClass', axis = 1).to_csv('/data/click_bait_detect/all_features_label_exceptImages.csv', index = False)


# In[30]:


data = pd.read_csv('/data/click_bait_detect/train_validation_truth_instances.csv')
data.columns


# In[31]:


da['postText_polarity'] = data['postText'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)
da['postText_subjectivity'] = data['postText'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)


# In[32]:


da['targetCaptions_polarity'] = data['targetCaptions'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)
da['targetCaptions_subjectivity'] = data['targetCaptions'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)


# In[33]:


da['targetDescription_polarity'] = data['targetDescription'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)
da['targetDescription_subjectivity'] = data['targetDescription'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)
da['targetParagraphs_polarity'] = data['targetParagraphs'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)
da['targetParagraphs_subjectivity'] = data['targetParagraphs'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)
da['targetTitle_polarity'] = data['targetTitle'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)
da['targetTitle_subjectivity'] = data['targetTitle'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)


# In[34]:


pos_di = {}
tagdict = load('help/tagsets/upenn_tagset.pickle')
for pos in list(tagdict.keys()):
	pos_di[pos] = []
for snt in data['postText']:
	di = Counter([j for i,j in pos_tag(word_tokenize(snt))])
	for pos in list(tagdict.keys()):
		pos_di[pos].append(di[pos])

da = pd.concat([da,pd.DataFrame(pos_di)], axis = 1)
#number of stop words
stp_wds = set(stopwords.words('english'))
da['postText_number_of_stop_words'] = data['postText'].apply(lambda x: len(stp_wds.intersection(word_tokenize(str(x)))))

#number of punctations
da['postText_num_of_unique_punctuations'] = data['postText'].apply(lambda x : len(set(x).intersection(set(string.punctuation))))


# In[35]:


# SAVE THE DATAFRAME
list(da.columns)


# In[36]:


da.to_csv('/data/click_bait_detect/all_features_label_exceptImages_v2_with_pos_polarity.csv', index = False)


# # Begin from here

# In[2]:


#da = pd.read_csv('/data/click_bait_detect/all_features_label_exceptImages_v2_with_pos_polarity.csv')
#da = pd.read_csv('/data/click_bait_detect/all_features_label_exceptImages_v2_with_pos_polarity_wmd.csv')
da = pd.read_csv('/data/click_bait_detect/all_features_label_exceptImages_v3_with_pos_polarity_wmd_extrafeatures.csv')
for i in ['number_of_images', 'number_of_targetKeyWords']:
    if i in da.columns:
        del da[i]
da = da[~da.isin([np.nan, np.inf, -np.inf]).any(1)]


# In[3]:


X = da.drop(['truthClass', 'truthClass_numeric'], axis =1)
y = da['truthClass_numeric']


# In[139]:


#ll = list(X.columns)
#X.columns = [str(j)+"_"+str(i) for i,j in zip(range(len(ll)),ll) if str(j)!='truthClass']


# In[4]:


X.shape#21997,106 #(21997, 369)


# In[50]:


y.shape


# In[5]:


#import pandas_profiling 
#profile = pandas_profiling.ProfileReport(X)
#rejected_variables = profile.get_rejected_variables(threshold=0.9)
#profile.to_file(outputfile="pandas_profiling_X_train_valid.html")


# ## Highly correlated features

# In[4]:


rejected_variables = [')',
 'sim_posttext_targetKeywords',
 'sim_targetTitle_targetCaptions',
 'sim_targetTitle_targetKeywords',
 'targetKeywords_golve_dim16',
 'targetKeywords_golve_dim26',
 'targetKeywords_golve_dim32',
 'targetKeywords_golve_dim33',
 'targetKeywords_golve_dim35',
 'wmd_postText_cleaned_splitted_targetKeywords_cleaned_splitted',
 'wmd_targetTitle_cleaned_splitted_targetKeywords_cleaned_splitted']


# ## Having more than 90% entries as zeros features
# 
# $ has 18333 / 98.0% zeros Zeros <br>
# EX has 18543 / 99.1% zeros Zeros  <br>
# FW has 18598 / 99.4% zeros Zeros  <br>
# JJR has 18096 / 96.7% zeros Zeros <br>
# JJS has 17791 / 95.1% zeros Zeros <br>
# NNPS has 18158 / 97.1% zeros Zeros <br>
# RBR has 18494 / 98.9% zeros Zeros <br>
# RBS has 18500 / 98.9% zeros Zeros <br>
# RP has 17695 / 94.6% zeros Zeros <br>
# UH has 18667 / 99.8% zeros Zeros <br>
# WDT has 18242 / 97.5% zeros Zeros <br>
# WP has 17632 / 94.3% zeros Zeros <br>
# WRB has 17782 / 95.1% zeros Zeros <br>

# In[5]:


variables_with_max_zero = ['$','EX', 'FW', 'JJR', 'JJS', 'NNPS', 'RBR', 'RBS', 'RP', 'UH', 'WDT', 'WP', 'WRB']
variables_with_constant_value = ['number_of_postText']


# In[6]:


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42,stratify = y)


# In[8]:


train_X.shape


# In[9]:


test_X.shape


# In[3]:


np.sqrt(383)


# # Function for plotting importance

# In[7]:


## function for plotting importance
def plot_importance(var,imp):
    plt.figure(figsize=(8,6))
    plt.barh(range(len(var)),imp,align='center')
    plt.yticks(range(len(var)),var)
    plt.xlabel('Importance of features')
    plt.ylabel('Features')
    plt.show()


# # Function to fit model on training data and gives accuracy and confusion matrix of train,validation and test data

# In[8]:


##Function to fit model on training data and gives accuracy and confusion matrix of train,validation and test data
def model(clf,train_X,train_y,test_X,test_y):
    clf.fit(train_X,train_y)  # fitting model
    #scoring data

    pred_tr=clf.predict(train_X)
    pred_test=clf.predict(test_X)
    fig, ax = plt.subplots(figsize=(8,6))
    print('\nTraining Accuracy:{}'.format(accuracy_score(train_y,pred_tr)))
    print('Mean Squared Error (MSE):{}'.format(mean_squared_error(train_y,pred_tr)))
    print('Training Confusion Matrix \n{}'.format(confusion_matrix(train_y,pred_tr)))
    print('Classification Report: \n{}'.format(classification_report(train_y,pred_tr)))
    pred_pr_tr=clf.predict_proba(train_X)[:,1]
    pred_pr_test=clf.predict_proba(test_X)[:,1]
    fpr,tpr,threshold=roc_curve(train_y,pred_pr_tr)
    ax.plot(fpr,tpr,label='Training AUC')
    print('\nAUC:{}\n'.format(auc(fpr,tpr)))

    print ('\nTest Accuracy:{}'.format(accuracy_score(test_y,pred_test)))
    print('Mean Squared Error (MSE):{}'.format(mean_squared_error(test_y,pred_test)))
    print ('Test Confusion Matrix \n{}'.format(confusion_matrix(test_y,pred_test)))
    print('Classification Report: \n{}'.format(classification_report(test_y,pred_test)))
    fpr,tpr,threshold=roc_curve(test_y,pred_pr_test)
    print ('\nAUC:{}\n'.format(auc(fpr,tpr)))
    ax.plot(fpr,tpr,label='Test AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.legend(loc='best')
    plt.show()


# # Logistic Regression

# In[30]:


print( '***********Logistic Regression**********')
lr_clf=LogisticRegression(random_state=1, class_weight = 'balanced')
train_XX = train_X.drop(rejected_variables+variables_with_max_zero+variables_with_constant_value, axis = 1)
test_XX = test_X.drop(rejected_variables+variables_with_max_zero+variables_with_constant_value, axis = 1)
model(lr_clf,train_XX,train_y,test_XX,test_y)

lr_train=lr_clf.predict(train_XX)
lr_test=lr_clf.predict(test_XX)
lr_pr_train=lr_clf.predict_proba(train_XX)[:,1]
lr_pr_test=lr_clf.predict_proba(test_XX)[:,1]
fpr, tpr, tresholds = roc_curve(train_y,lr_pr_train)

gini_train=2*(auc(fpr, tpr))-1
print('Gini (Train) : {}'.format(round(gini_train,2)))

fpr, tpr, tresholds = roc_curve(test_y,lr_pr_test)

gini_test=2*(auc(fpr, tpr))-1
print('Gini (Test) : {}'.format(round(gini_test,2)))


# In[32]:


ans = lr_clf.predict_proba(test_XX)[:,1]
ans


# In[34]:


use_df = test_XX
use_df['prediction'] = ans
use_df[use_df['prediction']==np.min(use_df['prediction'])]


# In[35]:


da[(da['number_of_targetParagraphs']==26) & (da['number_of_targetCaptions']==2)]#11283


# In[42]:


da[(da['number_of_targetParagraphs']==26) & (da['number_of_targetCaptions']==1)].iloc[27,:]#11283


# In[25]:


ddt = da[(da['number_of_targetParagraphs']==18) & (da['number_of_targetCaptions']==1)]


# In[28]:


pd.DataFrame([(a,b) for a,b in zip(list(ddt.columns),list(ddt.iloc[11,:]))]).to_csv('actual.csv', index = False)


# In[18]:


pickle.dump(lr_clf, open("/data/click_bait_detect/lr_clf_wmd_new_features_v2.pickle.dat", "wb"))


# In[42]:


len(train_XX.columns)


# In[47]:


list(use_df.columns)


# In[45]:


use_df[(use_df['number_of_targetCaptions']==3) & (use_df['number_of_targetParagraphs']==43)]


# In[41]:


list(train_XX.columns)


# # Random Forest

# In[9]:


print( '***********Random Forest**********')

rf_clf=RandomForestClassifier(n_estimators=20,max_depth=6,max_features='sqrt',random_state=1)
model(rf_clf,train_X,train_y,test_X,test_y)
rf_train=rf_clf.predict(train_X)
rf_test=rf_clf.predict(test_X)
rf_pr_train=rf_clf.predict_proba(train_X)[:,1]
rf_pr_test=rf_clf.predict_proba(test_X)[:,1]

var_imp=pd.DataFrame({'var':train_X.columns,'imp':rf_clf.feature_importances_})
var_imp.sort_values('imp',ascending=True,inplace=True)
vvimp = var_imp.sort_values('imp', ascending = False).head(30)
plot_importance(vvimp['var'],vvimp['imp'])
fpr, tpr, tresholds = roc_curve(train_y,rf_pr_train)

gini_train=2*(auc(fpr, tpr))-1
print('Gini (Train) : {}'.format(round(gini_train,2)))

fpr, tpr, tresholds = roc_curve(test_y,rf_pr_test)

gini_test=2*(auc(fpr, tpr))-1
print('Gini (Test) : {}'.format(round(gini_test,2)))


# In[57]:


pickle.dump(rf_clf, open("/data/click_bait_detect/rf_clf.pickle_wmd_new_features.dat", "wb"))


# In[10]:


var_imp.sort_values('imp', ascending = False).head(30)


# In[12]:


var_imp.sort_values('imp', ascending = False).head(30)['var'].values


# # XG Boost

# In[13]:


print( '***********XGBoost**********')

xgb_clf=XGBClassifier(n_estimators=6,learning_rate=0.01,max_depth=5,subsample=0.6,colsample_bytree= 0.6,reg_alpha= 10,seed=1)
model(xgb_clf,train_X,train_y,test_X,test_y)


xgb_train=xgb_clf.predict(train_X)
xgb_test=xgb_clf.predict(test_X)
xgb_pr_train=xgb_clf.predict_proba(train_X)[:,1]
xgb_pr_test=xgb_clf.predict_proba(test_X)[:,1]

var_imp=pd.DataFrame({'var':train_X.columns,'imp':xgb_clf.feature_importances_})
var_imp.sort_values('imp',ascending=True,inplace=True)


fpr, tpr, tresholds = roc_curve(train_y,xgb_pr_train)


gini_train=2*(auc(fpr, tpr))-1
print('Gini (Train) : {}'.format(round(gini_train,2)))
fpr, tpr, tresholds = roc_curve(test_y,xgb_pr_test)


gini_test=2*(auc(fpr, tpr))-1
print('Gini (Test) : {}'.format(round(gini_test,2)))


# In[14]:


var_imp.sort_values('imp', ascending = False).head(30)


# In[15]:


var_imp.sort_values('imp', ascending = False).head(30)['var'].values


# In[66]:


vvimp = var_imp.sort_values('imp', ascending = False).head(30)
plot_importance(vvimp['var'],vvimp['imp'])


# In[68]:


pickle.dump(xgb_clf, open("/data/click_bait_detect/xgb_clf.pickle_wmd_new_features.dat", "wb"))


# # LightGBM

# In[16]:


print( '***********LightGBM**********')

#lgb_clf=LGBMClassifier(n_estimators=16,learning_rate=0.01,max_depth=5,subsample=0.5,colsample_bytree= 0.5,reg_alpha= 10,seed=1)
lgb_clf=LGBMClassifier(n_estimators=25,learning_rate=0.01,max_depth=7,subsample=0.2,colsample_bytree= 0.3,reg_alpha= 5,seed=1)
model(lgb_clf,train_X,train_y,test_X,test_y)


lgb_train=lgb_clf.predict(train_X)
lgb_test=lgb_clf.predict(test_X)
lgb_pr_train=lgb_clf.predict_proba(train_X)[:,1]
lgb_pr_test=lgb_clf.predict_proba(test_X)[:,1]

var_imp=pd.DataFrame({'var':train_X.columns,'imp':lgb_clf.feature_importances_})
var_imp.sort_values('imp',ascending=True,inplace=True)


fpr, tpr, tresholds = roc_curve(train_y,lgb_pr_train)


gini_train=2*(auc(fpr, tpr))-1
print('Gini (Train) : {}'.format(round(gini_train,2)))
fpr, tpr, tresholds = roc_curve(test_y,lgb_pr_test)


gini_test=2*(auc(fpr, tpr))-1
print('Gini (Test) : {}'.format(round(gini_test,2)))


# In[17]:


var_imp.sort_values('imp', ascending = False).head(30)


# In[18]:


var_imp.sort_values('imp', ascending = False).head(30)['var'].values


# In[13]:


vvimp = var_imp.sort_values('imp', ascending = False).head(30)
plot_importance(vvimp['var'],vvimp['imp'])


# In[14]:


pickle.dump(lgb_clf, open("/data/click_bait_detect/lgb_clf.pickle_wmd_new_features.dat", "wb"))


# In[28]:


rf_top30_var = ['NNP',
       'wmd_postText_cleaned_splitted_targetParagraphs_cleaned_splitted',
       'postText_golve_dim26',
       'wmd_postText_cleaned_splitted_targetDescription_cleaned_splitted',
       'targetParagraphs_golve_dim28', 'postText_targetTitle_jaccard',
       'postText_golve_dim28', 'postText_golve_dim24',
       'targetTitle_golve_dim28', 'postText_golve_dim30',
       'targetParagraphs_golve_dim43', 'targetParagraphs_golve_dim23',
       'targetTitle_golve_dim24', 'targetTitle_golve_dim26',
       'postText_golve_dim43',
       'wmd_postText_cleaned_splitted_targetTitle_cleaned_splitted',
       'postText_golve_dim14', 'targetParagraphs_golve_dim9',
       'postText_golve_dim22', 'targetParagraphs_golve_dim26',
       'sim_posttext_targetTitle', 'targetTitle_golve_dim30',
       'sim_posttext_targetDescription',
       'postText_targetDescription_jaccard', 'postText_golve_dim9',
       'targetDescription_golve_dim12', 'sim_posttext_targetKeywords',
       'targetTitle_golve_dim19', 'postText_golve_dim11',
       'number_of_targetCaptions'][:10]


xgb_top30_var = ['wmd_postText_cleaned_splitted_targetParagraphs_cleaned_splitted',
       'targetParagraphs_golve_dim46', 'postText_golve_dim28',
       'postText_golve_dim24', 'postText_golve_dim26',
       'sim_posttext_targetKeywords', 'postText_has_wh_words',
       'wmd_postText_cleaned_splitted_targetDescription_cleaned_splitted',
       'postText_targetTitle_jaccard', 'targetParagraphs_golve_dim28',
       'postText_targetDescription_jaccard', 'postText_golve_dim30',
       'NNP', 'targetParagraphs_golve_dim43', 'postText_golve_dim14',
       'wmd_postText_cleaned_splitted_targetTitle_cleaned_splitted',
       'targetTitle_golve_dim22', 'targetDescription_golve_dim12',
       'postText_golve_dim43', 'postText_golve_dim29',
       'targetTitle_golve_dim14', 'targetDescription_golve_dim48',
       'targetDescription_golve_dim37', 'targetTitle_golve_dim32',
       'targetDescription_golve_dim36', 'targetDescription_golve_dim4',
       'sim_posttext_targetTitle', 'targetCaptions_golve_dim6',
       'targetTitle_golve_dim43', 'targetTitle_golve_dim38'][:10]


lgbm_top30_var = ['postText_targetDescription_jaccard', 'postText_golve_dim28',
       'postText_golve_dim14',
       'wmd_postText_cleaned_splitted_targetDescription_cleaned_splitted',
       'wmd_postText_cleaned_splitted_targetTitle_cleaned_splitted',
       'targetParagraphs_golve_dim28', 'postText_golve_dim30',
       'postText_targetTitle_jaccard', 'targetParagraphs_golve_dim46',
       'postText_has_wh_words', 'postText_golve_dim24',
       'targetParagraphs_golve_dim43', 'targetTitle_golve_dim26',
       'targetParagraphs_golve_dim23', 'postText_golve_dim26',
       'targetCaptions_golve_dim6', 'postText_golve_dim22',
       'targetCaptions_golve_dim18', 'targetParagraphs_golve_dim26',
       'postText_golve_dim43', 'targetTitle_golve_dim30',
       'sim_posttext_targetDescription', 'targetCaptions_golve_dim28',
       'targetParagraphs_golve_dim12', 'postText_golve_dim48',
       'wmd_postText_cleaned_splitted_targetKeywords_cleaned_splitted',
       'NNP', 'targetTitle_golve_dim37',
       'postText_num_of_unique_punctuations',
       'targetParagraphs_golve_dim40'][:10]


# In[29]:


common_rf_xgb_lgbm = set(rf_top30_var).intersection(set(xgb_top30_var)).intersection(set(lgbm_top30_var))
common_rf_xgb_lgbm


# In[33]:


set(rf_top30_var) - set(xgb_top30_var) - set(lgbm_top30_var)


# In[34]:


set(xgb_top30_var) -set(rf_top30_var) - set(lgbm_top30_var)


# In[35]:


set(lgbm_top30_var) - set(xgb_top30_var) -set(rf_top30_var)


# # Training seperate classifier for images

# #Please refer to
# http://172.29.75.251:1288/notebooks/clickbait_imageprocessing.ipynb

# # Creating API for deployment

# In[247]:


import re
word_tokenize
from textblob import Word
from gensim.models import KeyedVectors
from scipy import spatial
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from collections import Counter
import pickle


# In[335]:


list(train_X.columns)


# In[337]:


train_X['number_of_postText'].value_counts()


# ## INPUT

# In[150]:


#Always 1 drop it
print("Enter total number of tweets for the given post")
number_of_postText = 2

print("Enter number of pictures/captions in the target (i.e. news article)")
number_of_targetCaptions = 3

print("Enter number of paragraphs in the target (i.e. news article)")
number_of_targetParagraphs = 2

print("Please enter post(i.e. tweet(s))")
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


# In[159]:


#Vector dimension extraction
postText_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(postText)))]
targetTitle_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetTitle)))]
targetDescription_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetDescription)))]
targetParagraphs_cleaned_splitted = [Word(j).lemmatize().strip().lower() for i in eval(str(targetParagraphs)) for j in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetParagraphs)))]
targetKeywords_cleaned_splitted = str(targetKeywords).lower().split(',')
targetCaptions_cleaned_splitted = [Word(i).lemmatize().strip().lower() for i in word_tokenize(re.sub(r'([^\s\w]|_)+', '', str(targetCaptions)))]


# In[168]:


#Instead of writing file, compute mdedian
filename = '/data/click_bait_detect/glove.6B.50d.txt.word2vec'
model_gv = KeyedVectors.load_word2vec_format(filename, binary=False)
fp = open('/data/click_bait_detect/targetTitle_cleaned_splitted_glove_vectors', 'w')

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


# In[173]:


postText_vec = get_vector(postText_cleaned_splitted)
targetTitle_vec = get_vector(targetTitle_cleaned_splitted)
targetDescription_vec = get_vector(targetDescription_cleaned_splitted)
targetParagraphs_vec = get_vector(targetParagraphs_cleaned_splitted)
targetKeywords_vec = get_vector(targetKeywords_cleaned_splitted)
targetCaptions_vec = get_vector(targetCaptions_cleaned_splitted)


# In[176]:


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


# In[205]:


pos_di = {}
tagdict = load('help/tagsets/upenn_tagset.pickle')
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


# In[185]:


def get_similarity(rw1,rw2):
    result = 1 - spatial.distance.cosine(rw1, rw2)
    return result


# In[194]:


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


# In[211]:


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


# In[224]:


for vec in ['postText_vec', 'targetTitle_vec', 'targetDescription_vec', 'targetParagraphs_vec', 'targetKeywords_vec', 'targetCaptions_vec']:
    for itm,dim in zip(eval(vec),range(50)):
        test_df[vec[:-4]+'_golve_dim'+str(dim)] = [itm]


# In[251]:


lr_clf_loaded = pickle.load(open("/data/click_bait_detect/lr_clf.pickle.dat", "rb"))


# In[252]:


lr_clf_loaded.predict_proba(test_df)[:,1]


# In[248]:


rf_clf_loaded = pickle.load(open("/data/click_bait_detect/rf_clf.pickle.dat", "rb"))


# In[249]:


rf_clf_loaded.predict_proba(test_df)[:,1]


# In[ ]:


image UPLOAD
#image feature extract
https://github.com/sohomghosh/flask_process_from_html_form


# In[255]:


data.columns


# # Exploring ULMFit

# In[254]:


import fastai
from fastai import *
from fastai.text import * 


# In[285]:


data['postText_clean'] = data['postText'].str.replace("[^a-zA-Z]", " ")
data['label'] = data['truthClass'].replace(to_replace = {'no-clickbait': 0, 'clickbait': 1})


# In[288]:


# tokenization 
tokenized_doc = data['postText_clean'].apply(lambda x: x.split())

# remove stop-words 
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

# de-tokenization 
detokenized_doc = [] 
for i in range(len(data)): 
    t = ' '.join(tokenized_doc[i]) 
    detokenized_doc.append(t) 

data['text'] = detokenized_doc


# In[305]:


df_trn, df_val = train_test_split(data[['label', 'text']], stratify = data['label'], test_size = 0.4, random_state = 12)


# In[306]:


df_trn.head()


# In[307]:


df_val.head()


# In[308]:


data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")


# In[309]:


# Classifier model data
data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)


# In[310]:


learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.7)
# train the learner object with learning rate = 1e-2
learn.fit_one_cycle(1, 1e-2)


# In[311]:


learn.save_encoder('ft_enc')


# In[312]:


learn = text_classifier_learner(data_clas, drop_mult=0.7)
learn.load_encoder('ft_enc')


# In[313]:


learn.fit_one_cycle(1, 1e-2)


# In[314]:


# get predictions
preds, targets = learn.get_preds()

predictions = np.argmax(preds, axis = 1)
pd.crosstab(predictions, targets)


# In[324]:


predictions_list = predictions.tolist()


# In[325]:


targets_list = targets.tolist()


# In[330]:


train_y = targets_list
pred_tr = predictions_list
print('\nAccuracy:{}'.format(accuracy_score(train_y,pred_tr)))
print('Mean Squared Error (MSE):{}'.format(mean_squared_error(train_y,pred_tr)))
print('Training Confusion Matrix \n{}'.format(confusion_matrix(train_y,pred_tr)))
print('Classification Report: \n{}'.format(classification_report(train_y,pred_tr)))


# In[260]:


from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data


# In[261]:


df = pd.DataFrame({'label':dataset.target, 'text':dataset.data})


# In[262]:


df.head()


# In[263]:


df.shape


# In[264]:


df = df[df['label'].isin([1,10])]
df = df.reset_index(drop = True)
df.head()


# In[265]:


df.shape


# In[266]:


df['label'].value_counts()


# In[267]:


df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")


# In[268]:


df['text'].head()


# In[269]:


stop_words = stopwords.words('english')


# In[270]:


# tokenization 
tokenized_doc = df['text'].apply(lambda x: x.split())

# remove stop-words 
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

# de-tokenization 
detokenized_doc = [] 
for i in range(len(df)): 
    t = ' '.join(tokenized_doc[i]) 
    detokenized_doc.append(t) 

df['text'] = detokenized_doc


# In[271]:


df['text']


# In[296]:


df_trn, df_val = train_test_split(df, stratify = df['label'], test_size = 0.4, random_state = 12)


# In[297]:


df_trn.head()


# In[298]:


df_val.head()


# In[299]:


# Language model data
data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")


# In[300]:


data_lm


# In[274]:


# Classifier model data
data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)


# In[275]:


learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.7)
# train the learner object with learning rate = 1e-2
learn.fit_one_cycle(1, 1e-2)


# In[276]:


learn.save_encoder('ft_enc')


# In[277]:


learn = text_classifier_learner(data_clas, drop_mult=0.7)
learn.load_encoder('ft_enc')


# In[278]:


learn.fit_one_cycle(1, 1e-2)


# In[279]:


# get predictions
preds, targets = learn.get_preds()

predictions = np.argmax(preds, axis = 1)
pd.crosstab(predictions, targets)

