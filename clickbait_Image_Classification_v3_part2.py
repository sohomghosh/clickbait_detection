
# coding: utf-8

# In[1]:


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


# In[2]:


main_dir = "/data/click_bait_detect/"
train_dir = "media/"
path = os.path.join(main_dir,train_dir)


# In[3]:


image_data = pd.read_csv('/data/click_bait_detect/postMedia_single_truthClass.csv')
image_data['truthClass'] = image_data['truthClass'].replace({'no-clickbait':0, 'clickbait':1})
train_df, validate_df = train_test_split(image_data, test_size=0.33, random_state=42, stratify = image_data['truthClass'])
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[4]:


X = []
y = []
for p,category in zip(train_df['postMedia'], train_df['truthClass']):
    img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_COLOR)
    new_img_array = cv2.resize(img_array, dsize=(128,128))
    #plt.imshow(new_img_array,cmap="gray")
    #plt.show()
    X.append(new_img_array)
    y.append(category)


# In[6]:


X[0].shape


# In[7]:


X = np.array(X).reshape(-1,128,128,3)
y = np.array(y)

#Normalize data
X = X/255.0


# In[8]:


y


# In[9]:


X_valid = []
y_valid = []
for p,category in zip(validate_df['postMedia'], validate_df['truthClass']):
    img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_COLOR)
    new_img_array = cv2.resize(img_array, dsize=(128,128))
    #plt.imshow(new_img_array,cmap="gray")
    #plt.show()
    X_valid.append(new_img_array)
    y_valid.append(category)

    
X_valid = np.array(X_valid).reshape(-1,128,128,3)
y_valid = np.array(y_valid)

#Normalize data
X_valid = X_valid/255.0  


# # 4-CNN

# In[11]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

cnn4 = Sequential()
cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X[0].shape))
cnn4.add(BatchNormalization())

cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))

cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.25))

cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))

cnn4.add(Flatten())

cnn4.add(Dense(512, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.5))

cnn4.add(Dense(128, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.5))

cnn4.add(Dense(1, activation='softmax'))

cnn4.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
cnn4.summary()


# In[ ]:


history4 = cnn4.fit(X, y,
          batch_size=256,
          epochs=10,
          verbose=1,
          validation_data=(X_valid, y_valid))


# In[15]:


score4 = cnn4.evaluate(X_valid, y_valid, verbose=0)
print('Test loss:', score4[0])
print('Test accuracy:', score4[1])


# In[18]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

accuracy = history4.history['acc']
val_accuracy = history4.history['val_acc']
loss = history4.history['loss']
val_loss = history4.history['val_loss']
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


# In[22]:


train_preds_class = cnn4.predict_classes(X)
valid_preds_class = cnn4.predict_classes(X_valid)


# In[24]:


train_preds = cnn4.predict_proba(X)
valid_preds = cnn4.predict_proba(X_valid)


# In[25]:


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

