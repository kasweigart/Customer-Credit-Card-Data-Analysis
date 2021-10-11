#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[5]:


import ipywidgets as widgets


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


from IPython.display import display


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


from sklearn.ensemble import RandomForestClassifier


# In[10]:


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[11]:


dataframe = pd.read_csv('credit_card_customers.csv')


# # Data Analysis of Customer Attrition from Alan Bank

# ### Welcome to the web application for the data analysis of customer attrition from Alan Bank. The intended purpose of this application is to provide insight for managers into the behaviors of their customers.

# ### Customer Credit Card Data
# #### This is a brief look at the first five customer records found in the customer credit card dataset. 

# In[12]:


dataframe.head()


# In[13]:


dataframe.info()


# In[14]:


cleansedDataframe = dataframe.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2','CLIENTNUM'], axis=1)


# ### Here is the revised customer dataset used in the data analysis.

# In[15]:


cleansedDataframe.head()


# In[16]:


cleansedDataframe.info()


# ### Sum of Customers by Attrition Flag

# In[17]:


sns.set_theme(palette='bright')
sns.countplot(x='Attrition_Flag',data=cleansedDataframe)


# ### Distribution of Customers by Age

# In[18]:


sns.histplot(cleansedDataframe['Customer_Age'],kde=True)


# ### Sum of Customers by Gender

# In[19]:


sns.countplot(x='Gender',data=cleansedDataframe)


# ### Sum of Customers by Dependent Count

# In[20]:


sns.histplot(cleansedDataframe['Dependent_count'])


# ### Sum of Customers by Education Level

# In[21]:


sns.histplot(cleansedDataframe['Education_Level'])


# ### Sum of Customers by Income Category

# In[22]:


sns.histplot(cleansedDataframe['Income_Category'])


# ### Sum of Customers by Card Category

# In[23]:


sns.countplot(x='Card_Category',data=cleansedDataframe)


# ### Distribution of Customer by Months on Book

# In[24]:


sns.histplot(cleansedDataframe['Months_on_book'],kde=True)


# ### Distribution of Customers by Credit Limit

# In[25]:


sns.histplot(cleansedDataframe['Credit_Limit'],kde=True)


# ### Sum of Customers by Months Inactive in a 12 Month Period

# In[26]:


sns.histplot(cleansedDataframe['Months_Inactive_12_mon'])


# ### Sum of Customers by Contacts Count in a 12 Month Period

# In[27]:


sns.histplot(cleansedDataframe['Contacts_Count_12_mon'])


# ### Distribution of Customers by Total Revolving Balance

# In[28]:


sns.histplot(cleansedDataframe['Total_Revolving_Bal'],kde=True)


# ### Distribution of Customers by Average Open to Buy

# In[29]:


sns.histplot(cleansedDataframe['Avg_Open_To_Buy'],kde=True)


# ### Distribution of Customers by Total Transaction Amount Change from Q4 to Q1

# In[30]:


sns.histplot(cleansedDataframe['Total_Amt_Chng_Q4_Q1'],kde=True)


# ### Distribution of Customers by Total Transaction Amount

# In[31]:


sns.histplot(cleansedDataframe['Total_Trans_Amt'],kde=True)


# ### Distribution of Customers by Total Transaction Count

# In[32]:


sns.histplot(cleansedDataframe['Total_Trans_Ct'],kde=True)


# ### Distribution of Customers by Total Transaction Count Change from Q4 to Q1

# In[33]:


sns.histplot(cleansedDataframe['Total_Ct_Chng_Q4_Q1'],kde=True)


# ### Distribution of Customers by Average Utilization Ratio

# In[34]:


sns.histplot(cleansedDataframe['Avg_Utilization_Ratio'],kde=True)


# In[35]:


dataTypes = cleansedDataframe.dtypes
categoricalCols = cleansedDataframe.dtypes[dataTypes == 'object'].index.tolist()
cleansedDataframe[categoricalCols] = cleansedDataframe[categoricalCols].apply(lambda a: a.astype('category'))


# ### Data Model Cleansing

# In[36]:


cleansedDataframe.info()


# In[37]:


attritionBinaryDict = {'Attrited Customer': 1, 'Existing Customer': 0}
cleansedDataframe['Attrition_Flag'] = cleansedDataframe['Attrition_Flag'].map(attritionBinaryDict)


# ### Map Attrition Flag Values to Ones and Zeros

# In[38]:


cleansedDataframe['Attrition_Flag']


# In[39]:


def oneHotEncoder(dataColumn, dataframe):
    dummyVars = pd.get_dummies(dataframe[[dataColumn]])
    encodedDataFrame = pd.concat([dataframe, dummyVars], axis=1)
    encodedDataFrame = encodedDataFrame.drop([dataColumn], axis=1)
    return encodedDataFrame


# In[40]:


X = cleansedDataframe.drop('Attrition_Flag', axis=1)
y = cleansedDataframe['Attrition_Flag']


# ### Encode All Categorical Values to Numbers

# In[41]:


encodeColumns = X.select_dtypes('category').columns.to_list()
for column in encodeColumns:
    X = oneHotEncoder(column, X)
X.info()


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[43]:


randomForestClassifier = RandomForestClassifier()
randomForestClassifier.fit(X_train, y_train)


# In[44]:


randomForestClassifierPredict = randomForestClassifier.predict(X_test)


# ### Supervised Classification Report

# In[45]:


button = widgets.Button(
    description='Generate Report',
    disabled=False,
    button_style='info'
)

display(button)

def printReport(button):
    button.close()
    return print(classification_report(y_test, randomForestClassifierPredict))

button.on_click(printReport)


# ### Supervised Classification Confusion Matrix

# In[46]:


def printMatrix(button3):
    button3.close()
    display = ConfusionMatrixDisplay.from_estimator(randomForestClassifier,X_test,y_test,cmap='YlGnBu')
    plt.title('Confusion Matrix')
    plt.xticks(ticks=[0,1],labels=['Existing','Attrited'])
    plt.yticks(ticks=[0,1],labels=['Existing','Attrited'])
    plt.grid(False)
    return display

button3 = widgets.Button(
    description='Generate Matrix',
    disabled=False,
    button_style='info'
)

button3.on_click(printMatrix)
button3


# ### Customer and Credit Card Attributes Ranked by Importance

# In[47]:


importantVars = pd.Series(randomForestClassifier.feature_importances_, index=X_train.columns)
importantVars.sort_values(ascending=False,inplace=True)

def printImportancePlot(button2):
    button2.close()
    plt.figure(figsize=(15,10))
    return importantVars.plot(kind='bar')

button2 = widgets.Button(
    description='Generate Plot',
    disabled=False,
    button_style='info'
)

button2.on_click(printImportancePlot)
button2


# In[ ]:




