#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle


# In[2]:


df = pd.read_csv('ipl.csv')


# In[3]:


df.head()


# In[4]:


#Removing Unwanted Columns
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler','striker','non-striker']
df.drop(labels=columns_to_remove,axis=1,inplace=True)


# In[5]:


df.head()


# In[6]:


df['bat_team'].unique()


# In[7]:


#Removing Teams which aren't present now
consistent_teams=['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Delhi Daredevils',
         'Sunrisers Hyderabad']


# In[8]:


df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]


# In[9]:


#Removing The First 5 overs of data in every match
df = df[df['overs']>=5.0]


# In[10]:


df.head()


# In[11]:


df.dtypes


# In[12]:


df['date'] = pd.to_datetime(df['date'])


# In[13]:


encoded_df = pd.get_dummies(data=df, columns=['bat_team','bowl_team'])
encoded_df['date'] = df['date'].astype('int64')
encoded_df = encoded_df.astype(int)


# In[14]:


encoded_df.head()


# In[15]:


encoded_df.columns


# In[16]:


#Rearranging the columns
encoded_df = encoded_df[['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5',
       'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
       'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad','total']]


# In[17]:


#Splitting the data into train and test set
X_train = encoded_df.drop(labels='total',axis=1)[encoded_df['date']<=2016]
X_test = encoded_df.drop(labels='total',axis=1)[encoded_df['date']>=2017]


# In[18]:


y_train = encoded_df[encoded_df['date']<=2016]['total'].values
y_test = encoded_df[encoded_df['date']>=2017]['total'].values


# In[19]:


X_train.drop(labels='date',axis=1,inplace=True)
X_test.drop(labels='date',axis=1,inplace=True)


# In[20]:


#Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[21]:


ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor = GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)


# In[22]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[23]:


prediction = ridge_regressor.predict(X_test)


# In[24]:


import seaborn as sns
sns.distplot(y_test-prediction)


# In[27]:


from sklearn import metrics
print('MAE: ',metrics.mean_absolute_error(y_test,prediction))
print('MSE: ',metrics.mean_squared_error(y_test,prediction))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[29]:


#Creating a pickle file 
filename = 'first_innings-score-lr-model-pkl'
pickle.dump(ridge_regressor,open(filename, 'wb'))


# In[ ]:




