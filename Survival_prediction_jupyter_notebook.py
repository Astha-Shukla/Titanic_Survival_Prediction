#!/usr/bin/env python
# coding: utf-8

# Importing dependencies

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# data collection and preprocessing

# In[3]:


#load the data from csv to pandas
titanic_data = pd.read_csv("train.csv")


# In[4]:


#printing first five rows
titanic_data.head()


# In[5]:


#num of rows and col
titanic_data.shape


# In[6]:


#getting some basic info
titanic_data.info()


# In[7]:


#no of missing value in each col
titanic_data.isnull().sum()


# Handling missing values

# In[8]:


#drop cabin col from dataframe

titanic_data = titanic_data.drop(columns='Cabin', axis=1)


# In[9]:


# Replacing missing values in age column with mean value

titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[10]:


#Finding the mode value of embarked col
print(titanic_data['Embarked'].mode())


# In[11]:


#Get index of S
print(titanic_data['Embarked'].mode()[0])


# In[12]:


#replacing missing val in embarked cool with mode val
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace = True)


# In[13]:


# now check no of missing value in each col after handling missing value
titanic_data.isnull().sum()


# Data Analysis --- 

# In[14]:


# Statistical measure

titanic_data.describe()


# In[15]:


# Finding the number of people survived and not survived

titanic_data['Survived'].value_counts()


# Data Visualisation

# In[16]:


# Making count plot for survived column

sns.countplot('Survived', data = titanic_data)


# In[17]:


titanic_data['Sex'].value_counts()


# In[18]:


# Making count plot for gender column

sns.countplot('Sex', data = titanic_data)


# In[19]:


# Number of survivors based on gender

sns.countplot('Sex', hue='Survived', data=titanic_data)


# In[20]:


# Making count plot for pclass column

sns.countplot('Pclass', data = titanic_data)


# In[21]:


# Number of survivors based on Pclass which represent class of tickets.

sns.countplot('Pclass', hue='Survived', data=titanic_data)


# Encoding the categorical column into numerical form

# In[22]:


titanic_data['Sex'].value_counts()


# In[23]:


titanic_data['Embarked'].value_counts()


# In[28]:


# Converting categorical col
titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}}, inplace = True)


# In[29]:


titanic_data.head()


# Separating features column and target column 

# In[32]:


X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis = 1)
Y = titanic_data['Survived']


# In[33]:


print(X)


# In[34]:


print(Y)


# Splitting data into training data and testing data

# In[35]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[36]:


print(X.shape, X_train.shape, X_test.shape)


# In[37]:


print(Y.shape, Y_train.shape, Y_test.shape)


# Model Training

# Logistic Regresion Model

# In[38]:


model = LogisticRegression()


# In[41]:


# training logistic regression model with training data
model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[42]:


# Accuracy on training data 

X_train_prediction = model.predict(X_train)


# In[43]:


#Predicted value of survival

print(X_train_prediction)


# In[45]:


#Comparing correct value between Y_train and predicted value
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy_Score of training data :', training_data_accuracy)


# In[46]:


# Accuracy on test data 

X_test_prediction = model.predict(X_test)


# In[47]:


#Predicted value of survival

print(X_test_prediction)


# In[48]:


#Comparing correct value between Y_test and predicted value
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy_Score of test data :', test_data_accuracy)


# In[ ]:




