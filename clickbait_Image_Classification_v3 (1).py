
# coding: utf-8

# In[5]:


#!pip install opencv-python


# ## Reference: https://www.kaggle.com/ruchibahl18/cats-vs-dogs-basic-cnn-tutorial

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from sklearn.metrics import accuracy_score,roc_curve,classification_report,confusion_matrix,precision_recall_curve,auc, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


# In[3]:


main_dir = "/data/click_bait_detect/"
train_dir = "media/"
path = os.path.join(main_dir,train_dir)


# In[6]:


image_data = pd.read_csv('/data/click_bait_detect/postMedia_single_truthClass.csv')
image_data['truthClass'] = image_data['truthClass'].replace({'no-clickbait':0, 'clickbait':1})
train_df, validate_df = train_test_split(image_data, test_size=0.33, random_state=42, stratify = image_data['truthClass'])
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[7]:


train_df.head()


# In[9]:


train_df['truthClass'].value_counts(normalize = True)


# In[10]:


validate_df['truthClass'].value_counts(normalize = True)


# In[11]:


X = []
y = []
for p,category in zip(train_df['postMedia'], train_df['truthClass']):
    img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img_array, dsize=(128,128))#(80, 80))
    #plt.imshow(new_img_array,cmap="gray")
    #plt.show()
    X.append(new_img_array)
    y.append(category)


# In[12]:


#X = np.array(X).reshape(-1, 80,80,1)
X = np.array(X).reshape(-1,128,128,1)
y = np.array(y)

#Normalize data
X = X/255.0


# In[15]:


y


# In[16]:


X_valid = []
y_valid = []
for p,category in zip(validate_df['postMedia'], validate_df['truthClass']):
    img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img_array, dsize=(128,128))#(80, 80))
    #plt.imshow(new_img_array,cmap="gray")
    #plt.show()
    X_valid.append(new_img_array)
    y_valid.append(category)

    
#X_valid = np.array(X).reshape(-1, 80,80,1)
X_valid = np.array(X_valid).reshape(-1,128,128,1)
y_valid = np.array(y_valid)

#Normalize data
X_valid = X_valid/255.0  


# In[91]:


model.reset_states()
model = Sequential()

# Adds a densely-connected layer with 64 units to the model:
model.add(Conv2D(5,(3,3), activation = 'relu', input_shape = X.shape[1:]))#10
model.add(MaxPooling2D(pool_size = (2,2)))

# Add another:
#model.add(Conv2D(10,(3,3), activation = 'relu'))#comment
#model.add(MaxPooling2D(pool_size = (2,2)))#comment

#model.add(Flatten())
#model.add(Dense(5, activation='relu'))#10

model.add(Flatten())#Extra add
# Add a softmax layer with 10 output units:
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[92]:


earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, callbacks=[earlystop])


# In[93]:


train_preds_class = model.predict_classes(X)
valid_preds_class = model.predict_classes(X_valid)


# In[ ]:


#predictions = model.predict(X_valid)
#Then round off


# In[94]:


train_preds = model.predict_proba(X)
valid_preds = model.predict_proba(X_valid)


# In[95]:


ytrain = y
yvalid = y_valid

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


# # 1-Conv CNN

# In[64]:


cnn1 = Sequential()
cnn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Dropout(0.2))

cnn1.add(Flatten())

cnn1.add(Dense(128, activation='relu'))
cnn1.add(Dense(1, activation='softmax'))

cnn1.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
cnn1.summary()


# In[65]:


from keras.preprocessing.image import ImageDataGenerator
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(X, y, batch_size=256)
val_batches = gen.flow(X_valid, y_valid, batch_size=256)


# In[80]:


history1 = cnn1.fit_generator(batches, steps_per_epoch=train_df.shape[0]//256, epochs=50,
                    validation_data=val_batches, validation_steps=validate_df.shape[0]//256, use_multiprocessing=True)


# In[81]:


score1 = cnn1.evaluate(X_valid, y_valid, verbose=0)
print('Test loss:', score1[0])
print('Test accuracy:', score1[1])


# In[82]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

accuracy = history1.history['acc']
val_accuracy = history1.history['val_acc']
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[83]:


train_preds_class = cnn1.predict_classes(X)
valid_preds_class = cnn1.predict_classes(X_valid)


# In[84]:


train_preds = cnn1.predict_proba(X)
valid_preds = cnn1.predict_proba(X_valid)


# In[85]:


ytrain = y
yvalid = y_valid

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

