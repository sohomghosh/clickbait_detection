
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from IPython import get_ipython
import seaborn as sns
from sklearn.feature_selection import RFECV,RFE,VarianceThreshold
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_curve,classification_report,confusion_matrix,precision_recall_curve,auc
from sklearn.preprocessing import StandardScaler,scale,Normalizer
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,cross_val_score
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale,StandardScaler,RobustScaler,Normalizer
from sklearn.feature_selection import SelectKBest,f_regression,f_classif,SelectFromModel
import xgboost as xgb
import warnings
from xgboost import XGBClassifier
get_ipython().magic('matplotlib inline')
warnings.filterwarnings('ignore')


# In[2]:


data_features_label_1 = pd.read_csv('/data/click_bait_detect/data_timestamp_count_features_label.csv')
data_features_label_1.head()


# In[3]:


#Feature not yet used
'postMedia'


# In[4]:


feature = 'postText'


# In[5]:


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


# In[6]:


features_sent_sub.shape


# In[7]:


count_not_ok


# In[8]:


count_ok


# In[24]:


need_features = pd.DataFrame(features_sent_sub)
need_features.head()


# In[25]:


da = pd.concat([data_features_label_1,need_features], axis = 1)


# In[26]:


data_features_label_1.shape


# In[27]:


da.head()


# In[28]:


da['truthClass_numeric'] =  da['truthClass'].replace(to_replace = {'no-clickbait': 0, 'clickbait': 1})


# In[29]:


X = da.drop(['truthClass', 'truthClass_numeric'], axis =1)
y = da['truthClass_numeric']


# In[30]:


da.shape


# In[31]:


X.shape


# In[32]:


y.shape


# In[33]:


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)


# # Function for plotting importance

# In[18]:


## function for plotting importance
def plot_importance(var,imp):
    plt.figure(figsize=(8,6))
    plt.barh(range(len(var)),imp,align='center')
    plt.yticks(range(len(var)),var)
    plt.xlabel('Importance of features')
    plt.ylabel('Features')
    plt.show()


# # Function to fit model on training data and gives accuracy and confusion matrix of train,validation and test data

# In[19]:


##Function to fit model on training data and gives accuracy and confusion matrix of train,validation and test data
def model(clf,train_X,train_y,test_X,test_y):
    clf.fit(train_X,train_y)  # fitting model
    #scoring data

    pred_tr=clf.predict(train_X)
    pred_test=clf.predict(test_X)
    fig, ax = plt.subplots(figsize=(8,6))
    print ('\nTraining Accuracy:{}'.format(accuracy_score(train_y,pred_tr)))
    print ('Training Confusion Matrix \n{}'.format(confusion_matrix(train_y,pred_tr)))
    print('Classification Report: \n{}'.format(classification_report(train_y,pred_tr)))
    pred_pr_tr=clf.predict_proba(train_X)[:,1]
    pred_pr_test=clf.predict_proba(test_X)[:,1]
    fpr,tpr,threshold=roc_curve(train_y,pred_pr_tr)
    ax.plot(fpr,tpr,label='Training AUC')
    print ('\nAUC:{}\n'.format(auc(fpr,tpr)))

    print ('\nTest Accuracy:{}'.format(accuracy_score(test_y,pred_test)))
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

# In[20]:


print( '***********Logistic Regression**********')
lr_clf=LogisticRegression(random_state=1, class_weight = 'balanced')
model(lr_clf,train_X,train_y,test_X,test_y)

lr_train=lr_clf.predict(train_X)
lr_test=lr_clf.predict(test_X)
lr_pr_train=lr_clf.predict_proba(train_X)[:,1]
lr_pr_test=lr_clf.predict_proba(test_X)[:,1]
fpr, tpr, tresholds = roc_curve(train_y,lr_pr_train)

gini_train=2*(auc(fpr, tpr))-1
print('Gini (Train) : {}'.format(round(gini_train,2)))

fpr, tpr, tresholds = roc_curve(test_y,lr_pr_test)

gini_test=2*(auc(fpr, tpr))-1
print('Gini (Test) : {}'.format(round(gini_test,2)))


# # Random Forest

# In[34]:


print( '***********Random Forest**********')

rf_clf=RandomForestClassifier(n_estimators=200,max_depth=7,max_features='sqrt',random_state=1)
model(rf_clf,train_X,train_y,test_X,test_y)
rf_train=rf_clf.predict(train_X)
rf_test=rf_clf.predict(test_X)
rf_pr_train=rf_clf.predict_proba(train_X)[:,1]
rf_pr_test=rf_clf.predict_proba(test_X)[:,1]

var_imp=pd.DataFrame({'var':train_X.columns,'imp':rf_clf.feature_importances_})
var_imp.sort_values('imp',ascending=True,inplace=True)
plot_importance(var_imp['var'],var_imp['imp'])
fpr, tpr, tresholds = roc_curve(train_y,rf_pr_train)

gini_train=2*(auc(fpr, tpr))-1
print('Gini (Train) : {}'.format(round(gini_train,2)))

fpr, tpr, tresholds = roc_curve(test_y,rf_pr_test)

gini_test=2*(auc(fpr, tpr))-1
print('Gini (Test) : {}'.format(round(gini_test,2)))


# In[22]:


var_imp.sort_values('imp', ascending = False).head(20)


# # XG Boost

# In[23]:


print( '***********XGBoost**********')

xgb_clf=XGBClassifier(n_estimators=5,learning_rate=0.01,max_depth=3,subsample=0.6,colsample_bytree= 0.6,reg_alpha= 10,seed=1)
model(xgb_clf,train_X,train_y,test_X,test_y)


xgb_train=xgb_clf.predict(train_X)
xgb_test=xgb_clf.predict(test_X)
xgb_pr_train=xgb_clf.predict_proba(train_X)[:,1]
xgb_pr_test=xgb_clf.predict_proba(test_X)[:,1]



fpr, tpr, tresholds = roc_curve(train_y,xgb_pr_train)


gini_train=2*(auc(fpr, tpr))-1
print('Gini (Train) : {}'.format(round(gini_train,2)))
fpr, tpr, tresholds = roc_curve(test_y,xgb_pr_test)


gini_test=2*(auc(fpr, tpr))-1
print('Gini (Test) : {}'.format(round(gini_test,2)))

