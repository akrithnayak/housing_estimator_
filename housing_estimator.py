#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


dataset = pd.read_csv("data_csv.csv")


# In[4]:



# In[5]:


dataset.RM = dataset.RM.fillna(dataset['RM'].mean())


# In[6]:


# %matplotlib qt


# In[7]:


dataset.hist(bins=50, figsize=(20, 15))


# In[8]:


dataset['TAXRM'] = dataset['TAX']/dataset['RM']
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
# X_train, X_test, y_train, y_test = train_test_split()
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(dataset, dataset.CHAS):
    train_set = dataset.loc[train_index]
    test_set = dataset.loc[test_index]


# In[9]:


train_set


# In[10]:


test_set.size


# In[11]:


corr_matrix = dataset.corr()

corr_matrix['MEDV'].sort_values(ascending=False)


# In[12]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(dataset[attributes], figsize=(12, 8))


# In[13]:


dataset.plot(kind='scatter', x="RM", y="MEDV", alpha=0.8)


# In[ ]:





# In[14]:


corr_matrix = dataset.corr()

corr_matrix['MEDV'].sort_values(ascending=False)


# In[15]:


scatter_matrix(dataset[attributes], figsize=(12, 8))


# In[16]:


dataset.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[17]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(dataset)


# In[18]:


df = imputer.transform(dataset)


# In[19]:


df = pd.DataFrame(df, columns=dataset.columns)
type(df)


# In[20]:


train_set.info()


# In[21]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scalar', StandardScaler())
])
X = train_set.drop('MEDV', axis=1).copy()
y = train_set.MEDV.copy()
X_test = test_set.drop('MEDV', axis=1).copy()
y_test = test_set.MEDV.copy()


# In[22]:


df_train = pd.DataFrame(my_pipeline.fit_transform(X), columns=X.columns)
df_test = pd.DataFrame(my_pipeline.transform(X_test))


# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model = DecisionTreeRegressor()
# model = LinearRegression()
model = RandomForestRegressor()
model.fit(X, y)



# In[24]:


from sklearn.metrics import mean_squared_error
predictions = model.predict(X)


# In[25]:


lin_mse = mean_squared_error(y, predictions)
lin_rmse = np.sqrt(lin_mse)


# In[26]:


lin_rmse


# In[35]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)

rmse = np.sqrt(-scores)
scores


# In[28]:


rmse.mean()


# In[29]:


# Random forest = 3.3830190374884936
# Decision Tree  = 4.101769030476378
# Linear regressor = 4.329977771201731


# In[36]:


y_predictions_ = model.predict(X_test)
final_mse = mean_squared_error(y_test, y_predictions_)
final_rmse = np.sqrt(final_mse)


# In[37]:


final_rmse


# In[40]:


result_df = pd.DataFrame(y_predictions_, columns=['MEDV'])


# In[41]:


result_df.head(), y_test.head()


# In[42]:


df_train[:1]


# In[43]:


y_on = model.predict([[-0.412, 0.127323, 0.234, -0.2567, -1.6788, 0.2565, -.4357, 2.5678, -2.2345, -0.6789, -0.8976, 4.3456, -0.89765, -1.50789]])


# In[44]:


y_on


# In[45]:


y_predictions = model.predict(df_train[:1])


# In[46]:


y_predictions


# In[47]:


train_set.head()


# In[48]:


y_predictis = model.predict(df_train[5:])


# In[49]:


y_predictis


# In[50]:


y[:1]


# In[ ]:




