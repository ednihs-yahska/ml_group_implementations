#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
i#mport matplotlib.pyplot as plt
#import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


main_df = pd.read_csv('PA1_train.csv.xls')


# In[5]:


main_df.head()


# In[6]:


main_df.isnull().any()


# In[7]:


main_df.isna().any()


# In[8]:


main_df.columns


# In[9]:


print(len(main_df.columns))


# In[10]:


df = main_df.drop(['dummy','id'],1) #dropped dummy variable and id


# In[11]:


df['date'] = pd.to_datetime(main_df['date'])


# In[12]:


df['month'] = df['date'].apply(lambda x:x.month)


# In[13]:


df['year'] =  df['date'].apply(lambda x:x.year)
df['day'] = df['date'].apply(lambda x:x.day)


# In[14]:


df.columns


# In[15]:


df.describe()


# In[16]:


df = df.drop(['date'],1)
print(df.columns)


# In[17]:


print(len(df.columns))


# In[18]:


df.corr()


# In[19]:


before_normalization_corr=df.corr()['price']
(before_normalization_corr.sort_values())


# In[20]:



df.hist()


# In[21]:


features = before_normalization_corr.keys()
print(len(features))
imprtant_features = [x for x in features if abs(before_normalization_corr[x])>0.1]


# In[22]:


print(imprtant_features,len(imprtant_features))


# In[23]:


df = df[imprtant_features]
df.columns


# In[24]:


y_train = df['price'] #y_train will contain predictor values for training set


# In[25]:


y_train.value_counts().head()


# In[26]:


df = df.drop(['price'],1) #dropped price from training data as its not needed in it
df.columns


# In[27]:


df.head()


# In[28]:


df.bedrooms.value_counts()


# In[29]:


df.isnull().any()


# In[30]:


df.isna().any()


# In[31]:


(df['view'].mean())


# In[32]:


for feature in df.columns:
    mn = df[feature].min()
    rnge = df[feature].max() - df[feature].min()
    df[feature] = df[feature].apply(lambda x : (x-mn)/rnge)


# In[33]:


df.describe()


# In[34]:


df.head()


# In[35]:


ln=[]
for exp in range(-2,8):
    ln.append(10**(-exp))
learning_rate = np.array(ln)
print(learning_rate)


# In[36]:


df['dummy'] = main_df['dummy']


# In[37]:


df.head()


# In[38]:


df.shape[0]


# In[39]:


import math
def norm_cal(a,b):
    s = sum(a*b)
    #print('called',a)
    return s
#norm_cal(np.array([1,3]),np.array([2,4]))
        


# In[40]:


((df.iloc[0]))


# In[41]:


# mn = min(y_train)
# mx = max(y_train)
# y_train=y_train.apply(lambda x:(x-mn)/(mx-mn))


# In[42]:


y_train.head()


# In[43]:


final_weights = np.array([0]*12)
final_weights 


# In[ ]:



for lmbda in learning_rate[8:9]:
    weight = np.array(np.random.randn(1,12)[0])
    #print('weight is',weight)
    #print(weight)
    ##### calculate for current iteration
    number_of_iteration = 0
    c = 1
    while True:
        gradient = np.array(np.zeros(12))
        #print('this is the default gradient',gradient)
        for i in range(df.shape[0]):
            ith_data_instance = np.array(df.iloc[i])
            predicted_y = norm_cal(ith_data_instance,weight)
            #print('predicted y',predicted_y)
            actual_y = y_train[i]
            diff = predicted_y -actual_y
            weighted_sum = diff*ith_data_instance
            #print('weighted sum for each iteration',weighted_sum)
            gradient = gradient + weighted_sum
            #print('updated gradient',gradient)
        weight = weight - lmbda*gradient
        number_of_iteration = number_of_iteration + 1
        #print('gradient',gradient)
        #print('weight',weight)
        #break
        print('number of iteration',number_of_iteration,math.sqrt(norm_cal(gradient,gradient)))
        if(math.sqrt(norm_cal(gradient,gradient))<=0.5):
            final_weights = weight
            break
        if number_of_iteration==c*100:
            print('number of iteration',number_of_iteration,math.sqrt(norm_cal(gradient,gradient)))
            c = c + 1
    print(number_of_iteration)
        
            


# In[ ]:


#calculate sse:
sse_for_lambda = 0
for i in range(df.shape[0]):
    instance = np.array(df.iloc[i])
    predicted = norm_cal(instance,final_weight)
    actual = y_train[i]
    error = predicted - actual
    error = error**2
    sse_for_lambda = sse_for_lambda + error
    


# In[ ]:


sse_for_lambda 

