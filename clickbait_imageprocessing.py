
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt


# In[1]:


import pandas as pd


# In[14]:


import numpy as np
import pandas as pd
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
import warnings
from xgboost import XGBClassifier
from textblob import TextBlob
get_ipython().magic('matplotlib inline')
warnings.filterwarnings('ignore')
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from collections import Counter
from nltk.data import load


# In[13]:


get_ipython().magic('matplotlib inline')


# In[4]:


get_ipython().system('ls /data/click_bait_detect/clickbait17-validation-170630/media')


# In[6]:


get_ipython().system('ls /data/click_bait_detect/clickbait17-train-170331/media')


# In[8]:


path = '/data/click_bait_detect/clickbait17-train-170331/media/608610983897145344.jpg'
img = Image.open(path)


# In[10]:


#to convert to greyscale
img = img.convert('L')


# best down-sizing filter
img = img.resize((300, 300), Image.ANTIALIAS)


# In[16]:


np.array(img)


# In[19]:


np.array(img).shape


# In[82]:


DIR = '/data/click_bait_detect/clickbait17-train-170331/media/'
df = pd.DataFrame({})
for image in os.listdir(DIR):
    path = os.path.join(DIR, image)
    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((20, 20), Image.ANTIALIAS)
    features_df = pd.DataFrame(np.array(img).reshape(-1,400))
    features_df['id'] = image.split('.')[0]
    df = df.append(features_df)
    #print(image,np.array(img).reshape(-1,400)[0])


# In[83]:


df.head(10)


# In[84]:


df.shape


# In[85]:


df.to_csv('/data/click_bait_detect/train_images_features_id.csv', index = False)


# In[88]:


DIR = '/data/click_bait_detect/clickbait17-validation-170630/media/'
df = pd.DataFrame({})
for image in os.listdir(DIR):
    path = os.path.join(DIR, image)
    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((20, 20), Image.ANTIALIAS)
    features_df = pd.DataFrame(np.array(img).reshape(-1,400))
    features_df['id'] = image.split('.')[0]
    df = df.append(features_df)
    #print(image,np.array(img).reshape(-1,400)[0])


# In[89]:


df.head()


# In[90]:


df.to_csv('/data/click_bait_detect/validation_images_features_id.csv', index = False)


# In[91]:


get_ipython().system('cp /data/click_bait_detect/validation_images_features_id.csv /data/kaggle_data_ga_crp/')


# In[67]:


pd.DataFrame(np.array(img).reshape(-1,400))


# In[54]:


print(np.array(img).reshape(-1,90000)[0])


# In[18]:


plt.imshow(np.array(img))


# In[17]:


plt.imshow(np.array(img), cmap = 'gist_gray')


# In[93]:


train_images_features = pd.read_csv('/data/click_bait_detect/train_images_features_id.csv')
validation_images_features = pd.read_csv('/data/click_bait_detect/validation_images_features_id.csv')


# In[95]:


train_images_features.tail()


# In[97]:


validation_images_features.tail()


# In[99]:


validation_images_features['id'] = validation_images_features['id'].apply(lambda x : str(x).replace('photo_',''))


# In[103]:


train_iamges_features.append(validation_images_features).to_csv('/data/click_bait_detect/train_validation_images_features_id.csv', index = False)


# In[104]:


get_ipython().system(' cp /data/click_bait_detect/train_validation_images_features_id.csv /data/kaggle_data_ga_crp/')


# In[105]:


train_validation_images_features= train_images_features.append(validation_images_features)


# In[106]:


train_validation_images_features.shape


# In[107]:


train_validation_images_features.head()


# In[108]:


data = pd.read_csv('/data/click_bait_detect/train_validation_truth_instances.csv')


# In[118]:


data['postMedia']


# In[120]:


images = data['postMedia'].apply(lambda x: [str(i).replace('media/','').replace('photo_','').split('.')[0] for i in eval(x)])
labels = data['truthClass']


# In[124]:


img_label_df = pd.DataFrame({'image':images,'label':labels})
img_label_df.head()


# In[126]:


image_single_label_df = pd.DataFrame([[item]+list(img_label_df.loc[line,'label':]) for line in img_label_df.index for item in img_label_df.loc[line,'image']],columns=img_label_df.columns)


# In[127]:


image_single_label_df.shape


# In[128]:


image_single_label_df.to_csv('/data/click_bait_detect/image_single_label_df.csv', index = False)


# In[129]:


get_ipython().system('cp /data/click_bait_detect/image_single_label_df.csv /data/kaggle_data_ga_crp/')


# In[132]:


required_df = train_validation_images_features.merge(image_single_label_df, left_on = 'id', right_on = 'image', how = 'inner').drop('image', axis = 1)
required_df.head()


# In[133]:


required_df.to_csv('/data/click_bait_detect/train_validation_image_features_with_labels.csv', index = False)


# In[134]:


get_ipython().system('cp /data/click_bait_detect/train_validation_image_features_with_labels.csv /data/kaggle_data_ga_crp/')


# # Extracting faetures with VGG16

# In[ ]:


https://keras.io/applications/#extract-features-with-vgg16


# In[41]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = '/data/click_bait_detect/clickbait17-train-170331/media/608610983897145344.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)


# In[49]:


features.shape


# In[ ]:


##serial_no_of_image,7 rows, each row has 7 columns,each row each column has 512 values
##2^9 = 512


# In[ ]:


https://keras.io/applications/#fine-tune-inceptionv3-on-a-new-set-of-classes
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html    


# In[151]:


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

K.clear_session()
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


# In[152]:


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')


# In[153]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


# In[161]:


image_df = pd.read_csv('/data/click_bait_detect/postMedia_single_truthClass.csv')
train_generator = train_datagen.flow_from_dataframe(dataframe = image_df, directory='/data/click_bait_detect/media/', x_col='postMedia', y_col='truthClass', has_ext=True,batch_size=10)


# In[162]:


12690/10


# In[155]:


model.fit_generator(train_generator, steps_per_epoch = 10,epochs=5)


# In[141]:


for i, layer in enumerate(base_model.layers):
   print(i, layer.name)


# In[142]:


for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True


# In[143]:


from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])


# In[144]:


model.fit_generator(train_generator, steps_per_epoch = 10, epochs=5)


# In[163]:


pred = model.predict_generator(train_generator, steps=1269)
pred


# In[164]:


pred.shape


# In[167]:


from collections import Counter
Counter([[a,b].index(max([a,b])) for [a,b] in pred])


# In[ ]:


model.predict()


# In[54]:


data = pd.read_csv('/data/click_bait_detect/train_validation_truth_instances.csv')[['postMedia', 'truthClass']]
data.shape


# In[68]:


ddf = data
ddf['postMedia'] = ddf['postMedia'].apply(lambda x : eval(str(x)))


# In[69]:


image_df = pd.DataFrame([[item]+list(ddf.loc[line,'truthClass':]) for line in ddf.index for item in ddf.loc[line,'postMedia']],columns=ddf.columns)


# In[70]:


image_df.shape


# In[71]:


image_df.tail()


# In[93]:


image_df.to_csv('/data/click_bait_detect/postMedia_single_truthClass.csv', index = False)


# In[94]:


get_ipython().system('cp /data/click_bait_detect/postMedia_single_truthClass.csv /data/kaggle_data_ga_crp/')


# # Model prepare with image features

# In[3]:


images_features_df = pd.read_csv('/data/click_bait_detect/train_validation_image_features_with_labels.csv')
images_features_df.head()


# In[8]:


images_features_df['label_numeric'] =  images_features_df['label'].replace(to_replace = {'no-clickbait': 0, 'clickbait': 1})
X = images_features_df.drop(['label', 'label_numeric', 'id'], axis =1)
y = images_features_df['label_numeric']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42,stratify = y)


# In[5]:


## function for plotting importance
def plot_importance(var,imp):
    plt.figure(figsize=(8,6))
    plt.barh(range(len(var)),imp,align='center')
    plt.yticks(range(len(var)),var)
    plt.xlabel('Importance of features')
    plt.ylabel('Features')
    plt.show()


# In[16]:


##Function to fit model on training data and gives accuracy and confusion matrix of train,validation and test data
def model(clf,train_X,train_y,test_X,test_y):
    clf.fit(train_X,train_y)  # fitting model
    #scoring data

    pred_tr=clf.predict(train_X)
    pred_test=clf.predict(test_X)
    fig, ax = plt.subplots(figsize=(8,6))
    print ('\nTraining Accuracy:{}'.format(accuracy_score(train_y,pred_tr)))
    print('Mean Squared Error (MSE):{}'.format(mean_squared_error(train_y,pred_tr)))
    print ('Training Confusion Matrix \n{}'.format(confusion_matrix(train_y,pred_tr)))
    print('Classification Report: \n{}'.format(classification_report(train_y,pred_tr)))
    pred_pr_tr=clf.predict_proba(train_X)[:,1]
    pred_pr_test=clf.predict_proba(test_X)[:,1]
    fpr,tpr,threshold=roc_curve(train_y,pred_pr_tr)
    ax.plot(fpr,tpr,label='Training AUC')
    print ('\nAUC:{}\n'.format(auc(fpr,tpr)))

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


# In[9]:


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


# In[13]:


print( '***********Random Forest**********')

rf_clf=RandomForestClassifier(n_estimators=100,max_depth=5,max_features='sqrt',random_state=1)
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


# In[11]:


var_imp.sort_values('imp', ascending = False).head(50)


# In[12]:


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

