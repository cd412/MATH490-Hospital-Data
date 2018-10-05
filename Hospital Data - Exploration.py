
# coding: utf-8

# ## Data Exploration

# Import libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# Import dataset

# In[2]:


#df0 = pd.read_csv('hdd0313cy.csv', low_memory=False, nrows=1000) # all columns, limited rows


# In[4]:


#df0.shape
#(1544747, 135)


# In[2]:


# Specify the selected field and their datatypes 

type_arrival = {'sex': 'category',
                'er_mode': 'category',
                'admtype': 'category',
                'yoa': 'float16',
                'diag_adm': 'category',
                'pay_ub92': 'category',
                'provider': 'category', 
                'asource': 'category',
                'moa': 'float16',
                'age': 'float16',
                'race': 'category'}

type_target = {'tot': 'float64'}

type_departure = {'los': 'float64',  
                  'trandb': 'float64', 
                  'randbg': 'float64', 
                  'randbs': 'float64', 
                  'orr': 'float64',
                  'anes': 'float64',
                  'seq': 'float64', 
                  'lab': 'float64', 
                  'dtest': 'float64', 
                  'ther': 'float64', 
                  'blood': 'float64',
                  'phar': 'float64',
                  'psycchrg': 'float64',
                  'other': 'float64', 
                  'patcon': 'float64', 
                  'dispub92': 'category', 
                  'icu': 'float16', 
                  'ccu': 'float16',
                  'service': 'category',
                  'payer': 'category',
                 'er_fee': 'object',
                 'er_chrg': 'object'}

col_arrival = [*type_arrival]
col_departure = [*type_departure]
col_target = [*type_target]

usecols = col_arrival + col_departure + col_target
dtype = {}
for d in [type_arrival, type_departure, type_target]:
    for k, v in d.items():
        dtype[k] = v


# In[3]:


# import data using relevant columns
df1 = pd.read_csv('hdd0313cy.csv', 
                  usecols=usecols, 
                  dtype=dtype,
                 #nrows=100000,
                 )


# Remove newborn data

# In[4]:


df2 = df1.copy()
df2 = df2[df2.age > 0]
df2 = df2[df2.admtype != '4']
v = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'Z', 'A']
df2 = df2[df2.admtype.isin(v)]
df2.shape


# Remove columns that are not known upon admission. Remove rows that are missing relevant data.

# In[5]:


def getFullYear(y):
    '''Converts yoa from yy format to yyyy format'''
    if y == 0:
        return 2000
    elif y < 10:
        return float("200"+str(y))
    elif y < 25:
        return float("20"+str(y))
    elif y < 100:
        return float("19"+str(y))
    else:
        return y


# In[6]:


#strip leading zeros and convert to float
df2.er_fee = pd.to_numeric(df2.er_fee.str.lstrip('0')).astype('float64')
df2.er_chrg = pd.to_numeric(df2.er_chrg.str.lstrip('0')).astype('float64')


# In[7]:


df2.isnull().sum()


# In[8]:


df2 = df2[df2.sex != '9']
df2.sex.cat.remove_unused_categories()
print(df2.shape)
df2 = df2[df2.admtype != '9'] # rem
print(df2.shape)
df2 = df2[False == pd.isna(df2.admtype)]
print(df2.shape)
df2 = df2[False == pd.isna(df2.asource)]
print(df2.shape)
df2 = df2[False == pd.isna(df2.service)]
print(df2.shape)
df2 = df2[False == pd.isna(df2.race)]
print(df2.shape)
#df2 = df2[False == pd.isna(df2.dx1)]
print(df2.shape)
df2 = df2[False == pd.isna(df2.er_mode)]
print(df2.shape)
df2 = df2[False == pd.isna(df2.diag_adm)]
print(df2.shape)
df2['yoa'] = df2['yoa'].apply(getFullYear)
df2 = df2[df2.yoa >= 2005]
print(df2.shape)
df2 = df2[df2.age <= 100]
print(df2.shape)
df2.isnull().sum()


# In[9]:


# fill na with values

#df2.dx1 = df2.dx1.cat.add_categories('-1')
#df2.px1 = df2.px1.cat.add_categories('-1')


value = {'er_fee': 0.0,
        'er_chrg': 0.0,
#         'dx1': '-1',
#         'px1': '-1',
        }
df2 = df2.fillna(value)
df2.isnull().sum()


# Check datatypes

# In[10]:


print(df2.dtypes)


# Edit year of addmission collumn

# In[11]:


df2['doa'] = df2['yoa'] * 12 + (df2['moa']-1)
df2['doa'].head()


# In[12]:


df3 = df2.copy()
print(df3.shape)
#print(df3.columns)
#print(df3.dtypes)
#df3.describe()


# #### Export datasets to csv

# In[17]:


#df3.to_csv("df3.csv")


# In[13]:


df3_arrival = df3.filter(col_arrival + col_target)
print(df3_arrival.shape)
df3_arrival.head()


# In[19]:


#df3_arrival.to_csv("df3_arrival.csv")


# ## One-Hot Encoding

# In[14]:


df3_copy = df3.copy()


# In[16]:


fields = ['er_mode', 'admtype', 'diag_adm', 
          'pay_ub92', 'provider', 'asource', 'race']

df3_encoded = pd.get_dummies(df3_copy, columns=fields, prefix=fields)

print(df3_encoded.shape)


# In[22]:


#df3_encoded.head()


# In[23]:


#df_sample = df3_encoded.sample(10000)
#df_sample.to_csv("df_sample.csv")


# In[24]:


df3_encoded.to_csv("df3_encoded.csv")


# In[25]:


#df3_encoded.columns


# ## Binary Encoding

# In[15]:


import category_encoders as ce


# In[16]:


df3_copy = df3.copy()


# In[32]:


fields = ['er_mode', 'admtype', 'diag_adm', 'pay_ub92', 'provider', 'asource', 'race']


# In[33]:


encoder = ce.BinaryEncoder(cols=fields)
df_binary = encoder.fit_transform(df3_copy)


# In[34]:


df_binary.head(5)


# In[37]:


df_binary.to_csv("df_binary.csv")


# In[38]:


print(df_binary.columns)


# In[26]:


df3.filter(['age', 'tot']).groupby('age').mean().head() ##.plot(kind='bar', x='age', y='tot')


# In[27]:


df2.filter(['age', 'tot']).groupby('age').mean().plot(kind='line')
df2.filter(['age', 'tot']).groupby('age').count().plot(kind='line', color='blue')


# In[28]:


print(df2.filter(['sex', 'tot']).groupby('sex').mean().head())
print(df2.filter(['sex', 'tot']).groupby('sex').count().head())
df2.filter(['sex', 'tot']).groupby('sex').mean().plot(kind='bar')
df2.filter(['sex', 'tot']).groupby('sex').count().plot(kind='bar', color='blue')


# Graph components of costs

# In[29]:


X_var = 'yoa'

col_cost = ['trandb', 'orr', 'anes', 'seq', 'lab', 'dtest', 'ther', 'blood', 'phar',
             'psycchrg', 'other', 'patcon', X_var]

df_cost = df2.filter(col_cost)

mean_cost = df_cost.groupby(X_var).mean().sort_values(by=X_var)
#print(mean_cost)
ax = mean_cost.plot(kind='area',stacked='true')
ax.legend(loc="right", bbox_to_anchor=(1.3, 0.5), ncol=1)
plt.show()


# In[30]:


X_var = 'yoa'

col_cost = ['trandb', 'orr', 'anes', 'seq', 'lab', 'dtest', 'ther', 'blood', 'phar',
             'psycchrg', 'other', 'patcon', X_var]

df_cost = df2.filter(col_cost)

mean_cost = df_cost.groupby(X_var).mean().sort_values(by=X_var)
mean_cost = mean_cost.divide(mean_cost.sum(axis=1),axis=0)
#print(mean_cost)
ax = mean_cost.plot(kind='area',stacked='true')
ax.legend(loc="right", bbox_to_anchor=(1.3, 0.5), ncol=1)
plt.show()


# In[31]:


X_var = 'er_mode'

col_cost = ['trandb', 'randbg', 'orr', 'anes', 'seq', 'lab', 'dtest', 'ther', 'blood', 'phar',
             'psycchrg', 'other', 'patcon', X_var]

df_cost = df2.filter(col_cost)

mean_cost = df_cost.groupby(X_var).mean()
rr = df_cost.groupby(X_var).count()/df_cost.count()
ax1 = mean_cost.plot(kind='bar',stacked='true')
ax1.legend(loc="right", bbox_to_anchor=(1.3, 0.5), ncol=1)


plt.show()


# In[32]:


X_var = 'yoa'
y_var = 'tot'
print(df3.filter([X_var, y_var]).groupby(X_var).mean())
df3.filter([X_var, y_var]).groupby(X_var).mean().plot(kind='bar')
print(df3.filter([X_var, y_var]).groupby(X_var).count())
df3.filter([X_var, y_var]).groupby(X_var).count().plot(kind='bar', color='blue')

