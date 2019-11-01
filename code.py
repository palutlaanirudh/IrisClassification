#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading the libraries

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[2]:


# Loading dataset

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=columns)


# In[3]:

dataset

# In[5]:


dataset.shape


# In[6]:


dataset.size


# In[7]:


dataset.head(20)


# In[8]:


print(dataset.describe())


# In[9]:


print(dataset.groupby('sepal-length').size())


# In[10]:


print(dataset.groupby('class').size())


# In[97]:


dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[12]:


dataset.hist()
plt.show()


# In[13]:


scatter_matrix(dataset)
plt.show()


# In[22]:


# Splitting validation dataset

ar = dataset.values
print(ar.shape)


# In[98]:


# We use 80% of the dataset to train and 20% of it for validation
# We first set up the training data
A = ar[ : , 0:4] 
B = ar[ : , 4]

# A and B are for training the model

# In[119]:


validation_size = 0.2
seed = 7
A_train, A_val, B_train, B_val = model_selection.train_test_split(A, B, test_size=validation_size, random_state=seed)


# In[120]:


A_val.size


# In[121]:


A_train.size


# In[122]:


# Setting up the test harness
seed = 7
scoring = 'accuracy'


# In[123]:


# Setting up models to test and check accuracy
models = [
    ("LR", LogisticRegression(solver='liblinear', multi_class='ovr')),
    ("LDA", LinearDiscriminantAnalysis()),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier()),
    ("NB", GaussianNB()),
    ("SVM", SVC(gamma='auto'))
]

model = models[0][1]
print(model)
results = []


# In[124]:


# Cross validating to check the accuracy of each model and choosing model
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, A_train, B_train, scoring=scoring, cv=kfold)
    results.append(cv_results)
    names.append(name)
    print(name, "\t", cv_results.mean())


# We now find a model and fit it to our dataset. Once the model is fitted, we look to make predictions using it. Predictions such as what class a particular flower with certain characteristics belongs to, what chance it has of belonging to a particular class, etc.
# 
# For this particular case, since we observe the highest accuracy with the **SVM model**, we use it.
# 

# In[125]:


# Training the model
model = SVC(gamma='auto')
model.fit (A_train, B_train)


# In[126]:


predictions = model.predict(A_val)


# In[127]:


print(accuracy_score(B_val, predictions))


# In[128]:


print(classification_report(B_val, predictions))


# In[132]:


# We now save the model into a directory within the root folder
import pickle
fileloc = "finalised_model.sav"
with open(fileloc, 'wb') as file:
    pickle.dump(model, file)
    file.close()


# We can now predict the classification of any flower given its attributes as one of the 3 classes. This is based on the best model we have available, i.e, the one that gives the highest prediction accuracy. 
