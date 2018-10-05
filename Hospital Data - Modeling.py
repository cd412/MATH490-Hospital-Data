
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# Import dataset

# In[ ]:


#df3 = pd.read_csv('df3_encoded.csv', low_memory=False)
df3 = pd.read_csv('df_sample.csv', low_memory=False)
#df3 = pd.read_csv('df_binary.csv', low_memory=False)
#df3 = pd.read_csv('df3_arrival.csv', low_memory=False)


# In[ ]:


df3 = df3.drop(['payer', 'Unnamed: 0'], axis='columns')


# In[ ]:


df3.info()


# In[ ]:


binaries = {}
for c in [*df3.columns]:
    if df3[c].min() > -127 and df3[c].max() < 128:
        if df3[c].dtype == 'int64':
            binaries[c] = np.int8


# In[ ]:


df3 = df3.astype(binaries)


# In[ ]:


df3.info()


# ## Split data set into Train and Test

# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


y = df3.tot.astype(float)
X = df3.drop(columns=['tot'])
#print(y.dtypes)
#print('\n')
#print(X.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ## Decision Tree

# In[14]:


from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor


# In[15]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predicted = model.predict(X_test)


# In[16]:


mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)
print(mse)
print(r2)


# In[18]:


print(model.feature_importances_)


# ## AdaBoost

# In[13]:


from sklearn.ensemble import AdaBoostRegressor


# In[14]:


model = AdaBoostRegressor()
model.fit(X_train, y_train)
predicted = model.predict(X_test)


# In[15]:


mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)
print(mse)
print(r2)


# ## Boosting

# In[16]:


from sklearn.ensemble import GradientBoostingRegressor


# In[17]:


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
model = GradientBoostingRegressor(**params)
model.fit(X_train, y_train)
predicted = model.predict(X_test)


# In[18]:


mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)
print(mse)
print(r2)


# ## XGBoost

# In[10]:


import xgboost as xgb
from xgboost.sklearn import XGBRegressor


# In[11]:


# fit model to training data
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.3, 'loss': 'ls'}
model = XGBRegressor(**params)


# In[ ]:


xgb_param = model.get_xgb_params()
xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=5,
                  early_stopping_rounds=50)
print(cvresult)


# In[ ]:


model.set_params(n_estimators=cvresult.shape[0])


# In[ ]:


#Fit the algorithm on the data
model.fit(X_train, y_train)


# In[ ]:


#Predict training set:
predicted = model.predict(X_test)
test_predprob = alg.predict_proba(X_test)[:,1]


# In[ ]:


mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)
print(mse)
print(r2)


# In[ ]:


feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)[:10]
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')


# #### Tuning

# In[12]:


from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search


# In[20]:


param_test1 = {
 'max_depth': [2,4],
 'min_child_weight': [1,2]
}


# In[ ]:


gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.3, n_estimators=100, seed=0), 
 param_grid = param_test1, cv=5)
gsearch1.fit(X_train, y_train)
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)


# ## XGBoost 3

# In[36]:


# fit model to training data
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.1, 'loss': 'ls'}
model = XGBRegressor(**params)
model.fit(X_train, y_train)


# In[37]:


predicted = model.predict(X_test)


# In[38]:


mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)
print(mse)
print(r2)


# ## XGBoost 4

# In[33]:


# fit model to training data
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
model = XGBRegressor(**params)
model.fit(X_train, y_train)


# In[34]:


predicted = model.predict(X_test)


# In[35]:


mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)
print(mse)
print(r2)


# ## XGBoost 2

# In[33]:


# fit model to training data
params = {'n_estimators': 500, 'max_depth': 2, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
model = XGBRegressor(**params)
model.fit(X_train, y_train)


# In[34]:


predicted = model.predict(X_test)


# In[35]:


mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)
print(mse)
print(r2)


# ## Random Forest

# In[26]:


from sklearn.ensemble import RandomForestRegressor


# In[27]:


model = RandomForestRegressor()
model.fit(X_train, y_train)


# In[28]:


predicted = model.predict(X_test)


# In[29]:


mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)
print(mse)
print(r2)


# In[ ]:




