import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from pandas import DataFrame
import gc
import certifi
import urllib3
from urllib3.util import Retry
from PIL import Image
from IPython.display import Image
from IPython.core.display import HTML
import io

from scipy.sparse import csr_matrix

import os


with open('train.json') as datafile1:
    train_data=json.load(datafile1)
    
with open('test.json') as datafile2:
    test_data=json.load(datafile2)

with open('validation.json') as datafile3:
    valid_data=json.load(datafile3)
    
#Preparing training Dataframe
train_imgs_df = pd.DataFrame.from_records(train_data["images"])
train_labels_df = pd.DataFrame.from_records(train_data["annotations"])
train_df = pd.merge(train_imgs_df,train_labels_df,on="imageId",how="outer")
train_df["imageId"] = train_df["imageId"].astype(np.int)

#Preparing Validation Dataframe
valid_imgs_df = pd.DataFrame.from_records(valid_data["images"])
valid_labels_df = pd.DataFrame.from_records(valid_data["annotations"])
valid_df = pd.merge(valid_imgs_df,valid_labels_df,on="imageId",how="outer")
valid_df["imageId"] = valid_df["imageId"].astype(np.int)

#Preparing test Dataframe
test_df = pd.DataFrame.from_records(test_data["images"])
test_df["url"] = test_df["url"]
test_df["imageId"] = test_df["imageId"].astype(np.int)


print(valid_df.isnull().any(0))


train_image_arr = train_df[["imageId","labelId"]].apply(lambda x: [(x["imageId"],int(i)) for i in x["labelId"]], axis=1).tolist()
train_image_arr = [item for sublist in train_image_arr for item in sublist]
train_image_row = np.array([d[0] for d in train_image_arr]).astype(np.int)
train_image_col = np.array([d[1] for d in train_image_arr]).astype(np.int)

train_image_vals = np.ones(len(train_image_col))
train_image_mat = csr_matrix((train_image_vals, (train_image_row, train_image_col)))

train_image_mat.sum(0).astype(np.int)
train_image_mat.shape


row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])

csr_matrix((data, (row, col))).toarray()

def get_url(label_id, label_mat, df):
    data_col = train_image_mat.getcol(label_id)
    #tar_col = np.random.choice(np.where(data_col.toarray() == 1.0)[0],size=np.where(data_col.toarray() == 1.0)[0].shape[0]).tolist()
    tar_col = np.random.choice(np.where(data_col.toarray() == 1.0)[0],size=20).tolist()
    urls = df[df["imageId"].isin(tar_col)]["url"].tolist()
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ' '.join([u for u in urls])
    return images_list
    
for label in range(1,train_image_mat.shape[1]):
    url_list=get_url(label, train_image_mat, train_df)
    label_path=(os.path.realpath(''))+'\\train\\'+str(label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    print(label_path)
    i=1
    for url in url_list.split(' '):
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb.save((label_path)+'\\'+str(i)+'.jpeg', format='JPEG', quality=90)
        i=i+1
 
valid_image_arr = valid_df[["imageId","labelId"]].apply(lambda x: [(x["imageId"],int(i)) for i in x["labelId"]], axis=1).tolist()
valid_image_arr = [item for sublist in valid_image_arr for item in sublist]
valid_image_row = np.array([d[0] for d in valid_image_arr]).astype(np.int)
valid_image_col = np.array([d[1] for d in valid_image_arr]).astype(np.int)

valid_image_vals = np.ones(len(valid_image_col))
valid_image_mat = csr_matrix((valid_image_vals, (valid_image_row, valid_image_col)))

valid_image_mat.sum(0).astype(np.int)
valid_image_mat.shape

for label in range(1,valid_image_mat.shape[1]):
    url_list=get_url(label, train_image_mat, train_df)
    label_path=(os.path.realpath(''))+'\\validation\\'+str(label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    print(label_path)
    i=1
    for url in url_list.split(' '):
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb.save((label_path)+'\\'+str(i)+'.jpeg', format='JPEG', quality=90)
        i=i+1

####################################################################


#Part#1-- Builing the CNN

#Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
classifier=Sequential()

#Step#1 - Convolution Layer
classifier.add(Convolution2D(filters=32,kernel_size=(3,3), input_shape=(64, 64, 3),activation='tanh'))

#Step 2 - Pooling
classifier.add(MaxPool2D(pool_size= (2,2)))

#Adding second convolution layer
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),activation='sigmoid'))
classifier.add(MaxPool2D(pool_size= (2,2)))

#Adding third convolution layer
classifier.add(Convolution2D(filters=32,kernel_size=(3,3),activation='tanh'))
classifier.add(MaxPool2D(pool_size= (2,2)))

#Step 4 _ Flattening
classifier.add(Flatten())

#Step 5- Full Connection
classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'tanh'))
classifier.add(Dense(units = 228, kernel_initializer = 'uniform', activation = 'tanh'))

# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'mae', metrics = ['acc'])

#Part 2: Fitting CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen.flow


training_set = train_datagen.flow_from_directory('train',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='categorical')

validation_set = test_datagen.flow_from_directory('validation',target_size=(64, 64),
                                                         batch_size=32,
                                                         class_mode='categorical')

classifier.fit_generator(training_set,
                        steps_per_epoch=600,
                        epochs=2,
                        validation_data=validation_set,
                        validation_steps=200)


#Part #- Making Single Prediction

import numpy as np
from keras.preprocessing import image
test_image=image.load_img('train/1/2.jpeg',target_size=(64, 64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices

for i in range (1,result.shape[1]):
    if result[0][i]>0.02:
        print(i)
        
result[0].max()

test_df_final=test_df.copy()
test_df_final['result']=''

test_df_final['result'][1]
test_df_final.shape[0]
test_df_final['url'][1]

import urllib.request
for x in range(16792,test_df_final.shape[0]):
    with urllib.request.urlopen(test_df_final['url'][x]) as url:
        f = io.BytesIO(url.read())
        test_image=image.load_img(f,target_size=(64, 64))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        result=classifier.predict(test_image)
        for i in range (0,result.shape[1]):
            if result[0][i]>0.09:
                test_df_final['result'][x]=test_df_final['result'][x]+str(i)+' '

with urllib.request.urlopen('https://contestimg.wish.com/api/webimage/570f35feb2f4b95d223aa9b1-large') as url:
    f = io.BytesIO(url.read())
    test_image=image.load_img(f,target_size=(64, 64))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=classifier.predict(test_image)
    for i in range (0,result.shape[1]):
        if result[0][i]>0.09:
            print(i)

result=test_df_final[['imageId','result']]
result.to_csv('submission.csv',index=0)
