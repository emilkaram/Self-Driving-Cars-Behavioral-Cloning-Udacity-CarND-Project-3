
# coding: utf-8

# In[1]:


#Import Libraries
import numpy as np


# In[2]:


import os


# In[3]:


import sys


# In[7]:


import tensorflow as tf


# In[8]:


import csv


# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


import matplotlib.image as mpimg


# In[11]:


import cv2


# In[12]:


#variables


# In[13]:


lines = []


# In[14]:


images =[]


# In[15]:


measurments =[]


# In[20]:


#Load Data


# In[21]:


with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# In[13]:


len(lines)


# In[14]:


line[3]


# In[15]:


# Center images Data Preprocess (read image files , add orginal with the corsoping sterring angle measurments 
# and flipped with the -ve of corsoping sterring angle measurments)
# Note :data normilization and images cropping will be done in the CNN layers
for line in lines:
    source_path=line[0]
    filename = source_path.split('\\')[-1]
    current_path='img/' + filename
    img=cv2.imread(current_path)
    images.append(img)
    measure=float(line[3])
    measurments.append(measure)
    images.append(cv2.flip(img,1))
    measurments.append(measure * -1.0)
    


# In[16]:


# Left images Data Preprocess (read image files , add orginal with the corsoping sterring angle measurments 
# and flipped with the -ve of corsoping sterring angle measurments)
# Note :data normilization and images cropping will be done in the CNN layers
for line in lines:
    source_path=line[1]
    filename = source_path.split('\\')[-1]
    current_path='img/' + filename
    img=cv2.imread(current_path)
    images.append(img)
    measure=float(line[3])+0.2
    measurments.append(measure)
    images.append(cv2.flip(img,1))
    measurments.append(measure * -1.0)
    


# In[17]:


# Right images Data Preprocess (read image files , add orginal with the corsoping sterring angle measurments 
# and flipped with the -ve of corsoping sterring angle measurments)
# Note :data normilization and images cropping will be done in the CNN layers
for line in lines:
    source_path=line[2]
    filename = source_path.split('\\')[-1]
    current_path='img/' + filename
    img=cv2.imread(current_path)
    images.append(img)
    measure=float(line[3])-0.2
    measurments.append(measure)
    images.append(cv2.flip(img,1))
    measurments.append(measure * -1.0)
    


# In[68]:


#cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
#cv2.imshow('Image',images[0])
#plt.imshow(images[0])


# In[33]:


#images[0][0][0]


# In[17]:


source_path


# In[18]:


filename


# In[19]:


current_path


# In[20]:


#images


# In[21]:


#measurments


# In[ ]:


# Create Train , validation data and labels


# In[18]:


X_train =np.array(images)
y_train =np.array(measurments)


# In[75]:


X_train.shape


# In[76]:


y_train.shape


# In[77]:


y_train[100]


# In[ ]:


# Create the CNN layers inculding data normilization and images cropping


# In[29]:


from keras.layers import Flatten , Dense , Lambda ,Cropping2D ,MaxPool2D ,Dropout


# In[30]:


from keras.layers import Convolution2D


# In[31]:


from keras.optimizers import Adam


# In[32]:


from keras.models import Sequential


# In[33]:


model = Sequential()


# In[34]:


model.add(Lambda(lambda x: x/255-0.5 , input_shape=(160,320,3)))


# In[35]:


model.add(Cropping2D(cropping=((70,25),(0,0))))


# In[36]:


model.add(Convolution2D(24,5,5,subsample=(2,2), activation ="relu"))


# In[37]:


model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))


# In[38]:


model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))


# In[39]:


model.add(Convolution2D(64,3,3,activation="relu"))


# In[40]:


model.add(Convolution2D(64,3,3,activation="relu"))


# In[41]:


#model.add(Dropout(0.2))


# In[42]:


model.add(Flatten())


# In[43]:


model.add(Dense(100))


# In[44]:


model.add(Dense(50))


# In[45]:


model.add(Dense(10))


# In[46]:


model.add(Dense(1))


# In[47]:


model.compile(loss='mse' , optimizer='adam')


# In[48]:


model.summary()


# In[122]:


model.fit(X_train , y_train , validation_split=0.2 , shuffle=True ,epochs =1)


# In[123]:


model.save('model.h5')

