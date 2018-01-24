
# coding: utf-8

# In[9]:


import pandas as pd 
import numpy as np 
import cv2

from keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adagrad
from keras import backend as K
import os
from sklearn.model_selection import StratifiedKFold


# In[10]:


np.random.seed(42) 

data_dir = "/data"
output_dir = "/output"

if not os.path.exists(data_dir):
    data_dir = "data"
    
if not os.path.exists(output_dir):
    output_dir = "output"   


# In[11]:


def get_scaled_imgs(df):
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)

def get_more_images(imgs):
    
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
      
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
        
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
       
    more_images = np.concatenate((imgs,v,h))
    
    return more_images


# In[12]:


df_train = pd.read_json(data_dir + "/train.json")
df_test = pd.read_json(data_dir + "/test.json")
Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])
Xtr_more = get_more_images(Xtrain) 
Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))
Xtest = (get_scaled_imgs(df_test))


# In[14]:


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

if K.image_data_format() == 'channels_first':
    channel_axis = 1
else:
    channel_axis = 3

def getModel():
    
    model = Sequential()
    
    # Conv block 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
   
    # Conv block 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    # Conv block 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    #Conv block 4
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    # Flatten before dense
    model.add(GlobalMaxPooling2D())

    #Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))

    #Dense 2
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adagrad(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

getModel().summary()


# In[ ]:


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
for fold_n, (train, test) in enumerate(kfold.split(Xtr_more, Ytr_more)):
    print("FOLD nr: ", fold_n)
    model = getModel()

    batch_size = 128
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(output_dir + '/.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    model.fit(Xtr_more[train], Ytr_more[train], batch_size=batch_size, epochs=100, verbose=1
              , callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)
    
    model.load_weights(filepath = output_dir + '/.mdl_wts.hdf5')

    score = model.evaluate(Xtr_more[test], Ytr_more[test], verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    pred_test = model.predict(Xtest)

    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})

    submission.to_csv(output_dir + '/submission_' + str(fold_n) + '.csv', index=False)


# In[ ]:


stacked_0 = pd.read_csv(output_dir + '/submission_0.csv')
stacked_1 = pd.read_csv(output_dir + '/submission_1.csv')
stacked_2 = pd.read_csv(output_dir + '/submission_2.csv')
stacked_3 = pd.read_csv(output_dir + '/submission_3.csv')
stacked_4 = pd.read_csv(output_dir + '/submission_4.csv')
stacked_5 = pd.read_csv(output_dir + '/submission_5.csv')
stacked_6 = pd.read_csv(output_dir + '/submission_6.csv')
stacked_7 = pd.read_csv(output_dir + '/submission_7.csv')
stacked_8 = pd.read_csv(output_dir + '/submission_8.csv')
stacked_9 = pd.read_csv(output_dir + '/submission_9.csv')

sub = pd.DataFrame()
sub['id'] = stacked_0['id']
sub['is_iceberg'] = np.exp(np.mean(
    [
        stacked_0['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_1['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_2['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_3['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_4['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_5['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_6['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_7['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_8['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_9['is_iceberg'].apply(lambda x: np.log(x)),
    ], axis=0))

sub.to_csv(output_dir + '/submission.csv', index=False, float_format='%.6f')   

