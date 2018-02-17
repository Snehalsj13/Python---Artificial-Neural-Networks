
# coding: utf-8

# In[1]:


from IPython.display import display
import numpy as np
import pandas as pd #for structured data
import seaborn as sns #just like matplotlib

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16, 12)


# In[21]:


filepath = "pima-indians-diabetes.data.csv"


# In[22]:


columns = ['pregnant', 'plasma_glucose', 'blood_pressure', 'skin_fold', 'serum_insulin', 'bmi', 'pedigree', 'age', 'class']


# In[23]:


data = pd.read_csv(filepath, names=columns)


# In[15]:


display(data.head())
display(data.sample(5))


# In[16]:


data.dtypes


# In[17]:


data.describe()


# In[24]:


data.groupby('class').mean()


# In[25]:


display(data.groupby('class').mean())
display(data.groupby('class').std())


# In[26]:


get_ipython().set_next_input(u'corr = data.corr');get_ipython().magic(u'pinfo data.corr')


# In[27]:


corr = data.corr()
corr


# In[30]:


sns.heatmap(corr)


# In[31]:


sns.heatmap(corr, cmap=sns.cubehelix_palette())


# In[32]:


sns.heatmap(corr, cmap=sns.cubehelix_palette(as_cmap=True))


# In[33]:


sns.heatmap(corr, cmap=sns.cubehelix_palette(as_cmap=True), annot=True)


# In[34]:


data.skew()


# In[35]:


sns.distplot(data.blood_pressure)


# In[36]:


data['blood_pressure'] #query data from table


# In[39]:


data[['blood_pressure', 'serum_insulin']] #query data along with the coulmn name


# In[38]:


sns.distplot(data[['serum_insulin']])


# In[40]:


sns.pairplot(data[['plasma_glucose', 'blood_pressure', 'serum_insulin', 'class']], hue="class")


# In[41]:


array = data.values
X = array[:,0:8] #slice till 8th column
Y = array[:,8]


# In[42]:


#feature importance using Random Forest
from sklearn.ensemble import  ExtraTreeClassifier #sklearn = statistical manipulation
model = ExtraTreeClassifier() 
#random Forestation: 1000s of trees trained independently for taking decisions. At the end voting is done. 
#Best Ans= max votes(ensembling)
model.fit(X, Y)
print (model.feature_importances_)

