#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')


# ### Reading the Telecom-Churn.csv file

# In[2]:


tc=pd.read_csv("C:\\Users\\Vivek Nag Kanuri\\Downloads\\telecom-churn - telecom-churn.csv")


# In[3]:


tc.head()


# In[4]:


# shape of the csv file 
tc.shape


# In[5]:


# size of the csv file
tc.size


# In[6]:


tc.describe().T


# ### Features - Data Types 

# In[7]:


tc.info()


# ### Knowing about Null values 
# #### NOTE: As you see in the above cell all the features are cleaned, but only the Total Charges Column has Null values.

# In[8]:


tc.isnull().sum()


# ### While predicting , Data Cleaning is very important. The dataset is to be cleaned.
# #### The cleaning process for the null values in Total Charges column is as follows :
# ####    Instead of Removing The Null Values in Total Charges column ,I filled those Null values with median value .

# In[9]:


tc['TotalCharges']=tc['TotalCharges'].fillna(tc['TotalCharges'].median())


# In[10]:


tc['TotalCharges'].isnull().sum()

# See in the above cells we filled the null values with median value 
# so that there are no null values in the TotalCharges column


# ### Finding Unique values in each column

# In[11]:


col=['gender','MultipleLines','InternetService','Contract','PaymentMethod']
# Remaining columns are having values like 'yes' and 'no'
for i in col:
    print(i)
    print(tc[i].unique())
    print(tc[i].nunique())
    print()


# ### In the MultipleLines column there are data errors such as 
# #### there are 3 unique columns like 'yes', 'no' ,'no phone service' ,but the values  is nothing but the value 'No'.
# #### so we've to replace 'no phone service' as 'no'.

# In[12]:


tc['MultipleLines']=tc['MultipleLines'].replace(['No phone service'],'No')


# In[13]:


tc['MultipleLines'].unique()


# #### hence the data errors are corrected

# In[14]:


tc.head()


# In[15]:


# Correlation of this data is given by:

tc.corr()


# In[16]:


plt.figure(figsize=(12,6))
sns.heatmap(tc.corr(),annot=True,cmap='Blues')
plt.show()


# In[17]:


plt.figure(figsize=(5,6))
sns.countplot(x='SeniorCitizen',data=tc,palette='Greens_d')
plt.show()


# #### The above count plot tells about how many senior citizens are there .And there are less senior citizens.

# In[18]:


plt.figure(figsize=(12,7))
sns.scatterplot(x='tenure',y='MonthlyCharges',data=tc,hue='PaymentMethod')
plt.show()


# #### The above scatterplot shows Monthly Charges vs Tenure among various types of Payment Methods.

# In[19]:


px.scatter(tc,tc.MonthlyCharges,tc.TotalCharges,color='PaymentMethod',size='tenure')


# #### The above plot shows how Monthly Charges and Total Charges are varied in according to the Payment Method.

# In[20]:


px.histogram(tc,tc.MonthlyCharges,color='Churn')


# #### The above graph explains how Churn varies in according to Monthly Charges

# In[21]:


px.box(tc,tc.PaymentMethod,tc.TotalCharges,color='PaymentMethod')


# #### The above graph tells how Total Charges varies in according to Payment Method

# ### Prediction :-
# #### data preparation:

# In[22]:


tc.head()


# In[23]:


tc.drop('customerID',axis=1,inplace=True)


# In[24]:


tc.head()


# In[25]:


tc.drop('Partner',axis=1,inplace=True)
tc.drop('Dependents',axis=1,inplace=True)
tc.drop('StreamingMovies',axis=1,inplace=True)
tc.drop('StreamingTV',axis=1,inplace=True)
tc.drop('PaperlessBilling',axis=1,inplace=True)
tc.drop('SeniorCitizen',axis=1,inplace=True)


# #### Removing of unwanted columns.

# In[26]:


tc.head()


# ### Using Label Encoder to transform categorical values 

# In[27]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for col in tc.columns:
    if tc[col].dtype=='object':
        tc[col]=l.fit_transform(tc[col])


# In[28]:


tc.head()


# ### Splitting data for training and testing :

# In[29]:


x=tc.drop('Churn',axis=1)
y=tc['Churn']


# In[30]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=42)


# In[31]:


from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler(feature_range=(0,1))
xtrain=mms.fit_transform(xtrain)
xtest=mms.fit_transform(xtest)
xtrain=pd.DataFrame(xtrain)
xtest=pd.DataFrame(xtest)


# In[32]:


Res={'Model':[],'Accuracy':[],'Recall':[],'Precision':[],'F1':[]}


# In[33]:


Results=pd.DataFrame(Res)
Results.head()


# ### Importing Machine Learning Algorithms  :

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB 

lr=LogisticRegression()
dc=DecisionTreeClassifier()
rf=RandomForestClassifier()
et=ExtraTreesClassifier()
knn=KNeighborsClassifier(n_neighbors=5)
svm=SVC()
gnb=GaussianNB()

model=[lr,dc,rf,et,knn,svm,gnb]

for models in model:
    models.fit(xtrain,ytrain)
    
    ypred=models.predict(xtest)
    
    
    print('Model :',models)
    print('-----------------------------------------------------------------------------------------------------------------------')
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
    
    print('confusion matrix :',confusion_matrix(ytest,ypred))
    print('classification report:',classification_report(ytest,ypred))
    print('accuracy :',round(accuracy_score(ytest,ypred),2))
    print('precision :',round(precision_score(ytest,ypred),2))
    print('recall :',round(recall_score(ytest,ypred),2))
    print('f1 :',round(f1_score(ytest,ypred),2))
    print()
    R={'Model':models,
             'Accuracy':round(accuracy_score(ytest,ypred),2),
             'Recall':round(recall_score(ytest,ypred),2),
             'Precision':round(precision_score(ytest,ypred),2),
             'F1':round(f1_score(ytest,ypred),2)
            }
    Results=Results.append(R,ignore_index=True)


# In[35]:


Results


# ### The above are the Results of Various Models 

# ### Logistic Regression got High Accuracy with 81%
# ### Support vector machine got 2nd highest accuracy with 80%
