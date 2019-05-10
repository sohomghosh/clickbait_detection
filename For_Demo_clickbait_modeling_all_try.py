
# coding: utf-8

# In[1]:


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


# # Begin from here

# In[2]:


da = pd.read_csv('/data/click_bait_detect/all_features_label_exceptImages_v3_with_pos_polarity_wmd_extrafeatures.csv')
for i in ['number_of_images', 'number_of_targetKeyWords']:
    if i in da.columns:
        del da[i]
da = da[~da.isin([np.nan, np.inf, -np.inf]).any(1)]


# In[3]:


X = da.drop(['truthClass', 'truthClass_numeric'], axis =1)
y = da['truthClass_numeric']


# In[4]:


X.shape#21997,106 #(21997, 369)


# In[5]:


y.shape


# In[5]:


#import pandas_profiling 
#profile = pandas_profiling.ProfileReport(X)
#rejected_variables = profile.get_rejected_variables(threshold=0.9)
#profile.to_file(outputfile="pandas_profiling_X_train_valid.html")


# ## Highly correlated features

# In[6]:


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

# In[7]:


variables_with_max_zero = ['$','EX', 'FW', 'JJR', 'JJS', 'NNPS', 'RBR', 'RBS', 'RP', 'UH', 'WDT', 'WP', 'WRB']
variables_with_constant_value = ['number_of_postText']


# In[8]:


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42,stratify = y)


# In[9]:


train_X.shape


# In[10]:


test_X.shape


# In[11]:


np.sqrt(383)


# # Function for plotting importance

# In[12]:


## function for plotting importance
def plot_importance(var,imp):
    plt.figure(figsize=(8,6))
    plt.barh(range(len(var)),imp,align='center')
    plt.yticks(range(len(var)),var)
    plt.xlabel('Importance of features')
    plt.ylabel('Features')
    plt.show()


# # Function to fit model on training data and gives accuracy and confusion matrix of train,validation and test data

# In[13]:


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

# In[14]:


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


# In[18]:


pickle.dump(lr_clf, open("/data/click_bait_detect/lr_clf_wmd_new_features_v2.pickle.dat", "wb"))


# # Random Forest

# In[15]:


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


# In[16]:


var_imp.sort_values('imp', ascending = False).head(30)


# In[17]:


var_imp.sort_values('imp', ascending = False).head(30)['var'].values


# # XG Boost

# In[18]:


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


# In[19]:


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

