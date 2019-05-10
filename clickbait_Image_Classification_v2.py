
# coding: utf-8

# ## Reference https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification

# In[49]:


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


# In[23]:


image_data = pd.read_csv('/data/click_bait_detect/postMedia_single_truthClass.csv')


# In[24]:


image_data.shape


# In[25]:


image_data.head()


# In[13]:


image_data['postMedia'].nunique()


# In[12]:


get_ipython().system('ls /data/click_bait_detect/media/609297109095972864.jpg')


# In[14]:


image_data['truthClass'].value_counts()


# In[18]:


image_data['truthClass'].value_counts().plot.bar()
plt.show()


# In[26]:


FAST_RUN = True
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3 # RGB color


# ## Sample image

# In[21]:


sample = '609297109095972864.jpg'
image = load_img("/data/click_bait_detect/media/"+sample)
plt.imshow(image)
plt.show()


# In[16]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.optimizers import RMSprop


# In[17]:


import tensorflow as tf
from keras import backend as K

def auc_func(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# ## Model Architecture Type 1

# In[18]:


#Model architecture : https://i.imgur.com/ebkMGGu.jpg
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'mse', auc_func])

model.summary()


# In[27]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[28]:


earlystop = EarlyStopping(patience=10)


# In[29]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[30]:


callbacks = [earlystop, learning_rate_reduction]


# ## preparing training and test set

# In[31]:


train_df, validate_df = train_test_split(image_data, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[30]:


train_df['truthClass'].value_counts().plot.bar()
plt.show()


# In[31]:


validate_df['truthClass'].value_counts().plot.bar()
plt.show()


# In[32]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15


# In[33]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/data/click_bait_detect/media/", 
    x_col='postMedia',
    y_col='truthClass',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)


# In[34]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/data/click_bait_detect/media/", 
    x_col='postMedia',
    y_col='truthClass',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)


# In[37]:


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "/data/click_bait_detect/media/", 
    x_col='postMedia',
    y_col='truthClass',
    target_size=IMAGE_SIZE,
    class_mode='binary'
)
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# In[36]:


epochs=3 if FAST_RUN else 50
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# In[44]:


model.save_weights("/data/click_bait_detect/image_classification_model_type1.h5")


# In[37]:


model.load_weights("/data/click_bait_detect/image_classification_model_type1.h5")


# In[40]:


nb_samples = train_df.shape[0]
train_predict = model.predict_generator(train_generator, steps=np.ceil(nb_samples/batch_size))


# In[44]:


train_predict.shape


# In[46]:


nb_samples = validate_df.shape[0]
valid_predict = model.predict_generator(validation_generator, steps=np.ceil(nb_samples/batch_size))


# In[54]:


train_preds = train_predict
valid_preds = valid_predict


# In[47]:


train_preds_class = [1 if pred > .5 else 0 for pred in train_predict]
valid_preds_class = [1 if pred > .5 else 0 for pred in valid_predict]


# In[57]:


ytrain = train_df['truthClass'].replace({'no-clickbait':0, 'clickbait':1})
yvalid = validate_df['truthClass'].replace({'no-clickbait':0, 'clickbait':1})


# In[58]:


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


# In[45]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# ## Model Architecture Type 2

# In[47]:


model.reset_states()
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy','mse', auc_func])


# In[48]:


epochs=3 if FAST_RUN else 50
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# In[49]:


model.save_weights("/data/click_bait_detect/image_classification_model_type2.h5")


# In[50]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# ## Model Architecture Type 3 

# In[57]:


model.reset_states()

model = Sequential()

model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), activation='relu'))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))
    
model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.0001),
            metrics=['accuracy','mse', auc_func])


# In[58]:


epochs=2 if FAST_RUN else 50
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# In[59]:


model.save_weights("/data/click_bait_detect/image_classification_model_type3.h5")


# In[60]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

