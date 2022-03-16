#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df= pd.read_csv('D:\Assignments\Extra\Prac\P_1\winequality.csv',sep=',',na_values='')
df.head()


# In[2]:


df.info()


# In[3]:


df.shape


# In[4]:


x=df.drop(df.columns[0],axis=1)
x.head()


# In[5]:


x.describe().transpose()


# In[6]:


x['fixed acidity']=x['fixed acidity'].fillna(x['fixed acidity'].median())
x['volatile acidity']=x['volatile acidity'].fillna(x['volatile acidity'].median())
x['citric acid']=x['citric acid'].fillna(x['citric acid'].median())
x['residual sugar']=x['residual sugar'].fillna(x['residual sugar'].median())
x['chlorides']=x['chlorides'].fillna(x['chlorides'].median())
x['pH']=x['pH'].fillna(x['pH'].median())
x['sulphates']=x['sulphates'].fillna(x['sulphates'].median())
x.info()


# In[7]:


sns.countplot(x['quality'])


# In[8]:


x['quality'].unique()


# In[9]:


x['quality'].value_counts()


# In[10]:


x['quality']=x['quality'].replace(4,7)


# In[11]:


x['quality'].value_counts()


# In[12]:


#replacing all together
x['quality']=x['quality'].replace([8,3,9],7)


# In[13]:


x['quality'].value_counts()


# In[14]:


x=x.drop(x.columns[[-1]],axis=1)
print(x.head())
y=pd.DataFrame(df['quality'])
print(y.head())


# In[15]:


x.info()


# In[16]:


y.info()


# In[17]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# In[18]:


x.head()


# In[19]:


from sklearn.tree import DecisionTreeClassifier
clf_reg = DecisionTreeClassifier(criterion= 'entropy',random_state=2,max_depth=30)
clf_reg.fit(xtrain,ytrain)


from sklearn.ensemble import RandomForestClassifier
clf_reg1 = RandomForestClassifier(random_state=2,n_estimators=150)
clf_reg1.fit(xtrain,ytrain)


# In[20]:


ypred=clf_reg.predict(xtest)
ypred1=clf_reg1.predict(xtest)
print(ypred)
print(ypred1)


# In[21]:


from sklearn import metrics
accuracy=metrics.accuracy_score(ytest,ypred)
accuracy1=metrics.accuracy_score(ytest,ypred1)
print(accuracy)
print(accuracy1)


# In[22]:


print(clf_reg.score(xtrain,ytrain))
print(clf_reg1.score(xtrain,ytrain))


# In[23]:


from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
get_ipython().system('pip install pydotplus')
import pydotplus
#import graphviz

#xvar=x.drop(['quality','type'],axis=1)
feature_cols=x.columns
print(feature_cols)


# In[24]:



dot_data =StringIO()
export_graphviz(clf_reg,out_file=dot_data,
               filled=True,rounded=True,
               special_characters=True,
               feature_names=feature_cols)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('wine.png')
Image(graph.create_png())


# In[ ]:




