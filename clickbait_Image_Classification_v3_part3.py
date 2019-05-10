
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


# In[5]:


X = []
y = []
for p,category in zip(train_df['postMedia'], train_df['truthClass']):
    img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_COLOR)
    new_img_array = cv2.resize(img_array, dsize=(150,150))
    #plt.imshow(new_img_array,cmap="gray")
    #plt.show()
    X.append(new_img_array)
    y.append(category)


# In[6]:


X[0].shape


# In[7]:


X = np.array(X).reshape(-1,150,150,3)
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
    new_img_array = cv2.resize(img_array, dsize=(150,150))
    #plt.imshow(new_img_array,cmap="gray")
    #plt.show()
    X_valid.append(new_img_array)
    y_valid.append(category)

    
X_valid = np.array(X_valid).reshape(-1,150,150,3)
y_valid = np.array(y_valid)

#Normalize data
X_valid = X_valid/255.0  


# In[10]:


from keras.applications import VGG19

# Create the base model of VGG19
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape = X[0].shape, classes = 2)


# In[11]:


vgg19.summary()


# In[13]:


from keras.applications.vgg19 import preprocess_input

# Preprocessing the input 
X = preprocess_input(X)
X_valid = preprocess_input(X_valid)


# In[14]:


# Extracting features
train_features = vgg19.predict(np.array(X), batch_size=256, verbose=1)
val_features = vgg19.predict(np.array(X_valid), batch_size=256, verbose=1)


# In[15]:


print(train_features.shape, "\n",  val_features.shape)


# In[17]:


# Flatten extracted features
train_features = np.reshape(train_features, (8502, 4*4*512))
val_features = np.reshape(val_features, (4188, 4*4*512))


# In[18]:


from keras.layers import Dense, Dropout
from keras.models import Model
from keras import models
from keras import layers
from keras import optimizers

# Add Dense and Dropout layers on top of VGG19 pre-trained
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="softmax"))


# In[19]:


import keras

# Compile the model
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[24]:


# Train the the model
history = model.fit(train_features, y,
          batch_size=256,
          epochs=50,
          verbose=1,
          validation_data=(val_features, y_valid))


# In[27]:


score = model.evaluate(val_features, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[28]:


# plot the loss and accuracy
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()


# In[29]:


train_preds_class = model.predict_classes(train_features)
valid_preds_class = model.predict_classes(val_features)


# In[30]:


train_preds = model.predict_proba(train_features)
valid_preds = model.predict_proba(val_features)


# In[31]:


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

