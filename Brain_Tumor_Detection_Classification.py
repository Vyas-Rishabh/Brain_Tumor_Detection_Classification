#!/usr/bin/env python
# coding: utf-8

# # Brain Tumor Detection Classification

# ### Load Module

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ### Prepare/Collect Data

# In[2]:


import os

path = os.listdir('F:/A DATA SCIENCE 2023/Brain Tumor Detection Classification/Training/')
classes = {'no_tumor':0, 'pituitary_tumor':1}


# In[3]:


import cv2
X = []
Y = []
for cls in classes:
    pth = 'F:/A DATA SCIENCE 2023/Brain Tumor Detection Classification/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j,0)
        img = cv2.resize(img, (200, 200))
        X.append(img)
        Y.append(classes[cls])


# In[4]:


np.unique(Y)


# In[5]:


X = np.array(X)
Y = np.array(Y)


# In[6]:


pd.Series(Y).value_counts()


# In[7]:


X.shape


# ### Visualize Data

# In[8]:


plt.imshow(X[0], cmap='gray')


# ### Prepare Data

# In[9]:


X_updated = X.reshape(len(X), -1)
X_updated.shape


# ### Split Data

# In[10]:


xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=.20)


# In[11]:


xtrain.shape, xtest.shape


# ### Feature Scaling

# In[12]:


print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# ### Feature Selection : PCA

# In[13]:


from sklearn.decomposition import PCA


# In[14]:


print(xtrain.shape, xtest.shape)
pca = PCA(.98)
pca_train = xtrain
pca_test = xtest


# ### Train Model

# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[16]:


import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(pca_train, ytrain)


# In[17]:


sv = SVC()
sv.fit(pca_train, ytrain)


# ### Evaluation

# In[18]:


print("Training Score:", lg.score(pca_train, ytrain))
print("Testing Score:", lg.score(pca_test, ytest))


# In[19]:


print("Training Score:", sv.score(pca_train, ytrain))
print("Testing Score:", sv.score(pca_test, ytest))


# ### Prediction

# In[20]:


pred = sv.predict(pca_test)
np.where(ytest != pred)


# In[22]:


pred[36]


# In[23]:


ytest[36]


# ### Test Model

# In[28]:


dec = {0:'No Tumor', 1:'Positive Tumor'}


# In[31]:


plt.figure(figsize=(12,8))
p = os.listdir('F:/A DATA SCIENCE 2023/Brain Tumor Detection Classification/Testing/')
c=1
for i in  os.listdir('F:/A DATA SCIENCE 2023/Brain Tumor Detection Classification/Testing/no_tumor/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('F:/A DATA SCIENCE 2023/Brain Tumor Detection Classification/Testing/no_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[32]:


plt.figure(figsize=(12,8))
p = os.listdir('F:/A DATA SCIENCE 2023/Brain Tumor Detection Classification/Testing/')
c=1
for i in  os.listdir('F:/A DATA SCIENCE 2023/Brain Tumor Detection Classification/Testing/pituitary_tumor/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread('F:/A DATA SCIENCE 2023/Brain Tumor Detection Classification/Testing/pituitary_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1

