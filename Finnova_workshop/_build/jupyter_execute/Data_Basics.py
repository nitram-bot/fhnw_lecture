#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')


# 
# 1. Morning
#     - Data Modelling & Cross Validation
#     - data leakage & dependent data
#     - imbalanced data (example in python)
#     - study: Ebanking Fraud 
#     - Q&A
#     
#     <br>
#     
#     
# 2. Afternoon
#     - __Data Basics & historical perspective__
#     - Linear Regression
#     - Trees
#     - house prices (regression example in python)
#     - Clustering
#     - bonus: Hyperparameter Optimization and AutoML
#     - Q&A

# ## Titanic - Machine Learning from Disaster
# ### Predict survival on the Titanic and get familiar with ML basics
# [Home Page](https://www.kaggle.com/c/titanic)

# In[20]:


import pandas as pd
train = pd.read_csv("../data/titanic/train.csv")

example = train[['Pclass', 'Sex', 'Age', 'Cabin']]
example.head()


# ### continuous variables
#  - are always approximations, e.g. the age could be 22 years, 2 months, 1 week, 1 day, 10 hours, 2 minuts, 55 seconds, ...
#  - have an ordering, e.g. $22\,\text{years} < 38\,\text{years}$
#  - you can interpret differences: $38\,\text{years} - 22\,\text{years} = 16\,\text{years}$

# In[21]:


example[['Age']].head()


# ### categorical or discrete variables
#   - are always finite, e.g. Sex is most of the time binary and it's either female or male (I know that's not the best exambple - please do not confuse sex with gender)
#   - there is no ordering
#   - you can not interpret differences
#   - sex is a special case because it's also a __binary variable__; another example for discrete variables is marital status that can be __single, married, widowed, divorced, separated__

# In[22]:


example[['Sex']].head()


# ### ordinal variables
#   - are always finite, e.g the passenger-class is either 1, 2, or 3 but nothing in between
#   - it's still possible to have an ordering: $\text{p-class 1} > \text{p-class 2} > \text{p-class 3}$
#   - you can not interpret the differences: $\text{p-class 3} - \text{p-class 2} = \text{p-class 
#  1}$?

# In[23]:


example[['Pclass']].head()


# ## Processing of Variable
# ### Continuous Variables
# 
# We can easily transform a continuous variable into a ordinal variable by setting cut-points. E.g., we could say that all passengers younger than 2 years are renamed as 'Baby', all passengers between 2 years and 17 years as 'Child', etc..

# In[24]:


example['Age_binned'] =pd.cut(example.Age,bins=[0,2,17,65,99],labels=['Baby','Child','Adult','Elderly'])# .iloc[30:40]
example[['Age','Age_binned']][30:40]


# ### Categorical or discrete variables
# For mathematical models it's hard to work with categories as for example *male*, *female* or *Adult*, *Baby*, *Child*, etc..<br>
# This is why we have to turn them into categorical variables. This is done by adding new columns to the data, one for each category-level:

# In[25]:


nexample = pd.concat([example, pd.get_dummies(example['Age_binned'])], axis=1)
nexample[['Age','Age_binned','Baby', 'Child', 'Adult', 'Elderly']][30:40]


# This is called:<br>
#   - one-hot encoding
#   - sometimes this is also incorrectly called dummy encoding
#   - real __dummy encoding__ has one column less than one-hot encoding: The idea is, if its not *Child*, nor *Adult* or *Elderly*, then is must be *Baby* - so we do not need an extra column for *Baby*
#   
# Most intuitively the __real__ dummy-encoding can be seen with __sex__: even though there are two different categories, we just need one column

# In[26]:


nnexample = pd.concat([nexample, nexample[['Sex']].replace({"Sex": {'female':1, 'male': 0}}).rename(columns={'Sex': 'Sex2'})], axis=1)
nnexample[['Sex', 'Sex2']].head()


# ### ordinal data
#   - there are methods for ordinal data, e.g. ordinal regression
#   - most of the time ordinal variables are just treated as categorical variables

# ## Missing Data
# Data can be missing:
#   - at random
#   - systematically, i.e. the fact that the data is missing could bear some valuable information
#   
# For categorical and ordinal data, missing data is just another category.<br>
# For continuous variables there are several possibilities to deal with missing data. The most frequent ones are:<br>
#   - imputation by the mean (the mean-value of all non-missing values is taken)
#   - imputation by the median (the value with half of all values larger and half of all values smaller is taken)
#   - imputation by the mode (the most frequent value is taken)
#   
# However, since we do not know for sure, why data is missing it is often helpfull to keep track of it by creating a new indicator variable for missing values:
# 
# __First, we create the indicator variable:__

# In[27]:


nnexample['missing_Age'] = pd.isna(nexample['Age']).astype(int)
nnexample[['Age', 'missing_Age']][30:40]


# __Second, we impute missing values with the average Age:__

# In[28]:


nnexample['Age'] = nnexample['Age'].fillna(nnexample['Age'].mean())
nnexample[['Age', 'missing_Age']][30:40]


# ### Interactions
# Interactions are another important concept in linear modelling. Here, the effect of one variable on the dependent variable $y$ depends on the value of another variable.<br>
# In the example below we try to model the probability that a person buys a house. Of course, monthly income is an important variable and the higher it is, the more likely that said person will buy a house. Another important variable is marital status. Married people with children in the household tend strongly to buy houses, especially if their monthly income is high. On the other hand, singles, even if they have a high income, will tend not to buy a house.<br>
# So we see, the variable "monthly income" __interacts__ with the variable "marital status": the effect of the two variables together is more than the sum of the effects of the individual variables.

# In[29]:


import numpy as np
from statsmodels.graphics.factorplots import interaction_plot
import pandas as pd

income = np.random.randint(0, 2, size = 80) # low vs high
marital = np.random.randint(1, 4, size = 80) # single, married, married & kids

probability = np.random.rand(80) + income * np.random.rand(80) * marital
probability = (probability - np.min(probability))
probability = probability/np.max(probability)

marital = pd.Series(marital)
marital.replace(to_replace = {1:'single', 2:'married', 3:'marrid w kids'}, inplace =True)

income = pd.Series(income)
income.replace(to_replace = {0:'low', 1:'high'}, inplace = True)

fig = interaction_plot(income, marital, probability,
                       colors=['mediumorchid', 'cyan', 'fuchsia'], ms=10, xlabel='income',
                       ylabel='probability of buying a house',
                       legendtitle='marital status')


# ### Standardization / Normalization
# Sometimes we have to bring different variables into the same range. This is very important for Neural Networks, but also for other algorithms it can sometimes be benefitial. <br>
# Assume, we have the age of some passengers as in the following table:

# In[30]:


from IPython.display import display, display_html

passenger_age = train[['Name', 'Age']].dropna()[0:15]
mean_std = pd.DataFrame({'mean':[passenger_age['Age'].mean()], 'standarddeviation':[passenger_age['Age'].std()]})

df1_styler = passenger_age.style.set_table_attributes("style='display:inline'").set_caption('passengers and age')
df2_styler = mean_std.style.set_table_attributes("style='display:inline'").set_caption('mean value and standarddeviation')

display_html(df1_styler._repr_html_() + ' ' + df2_styler._repr_html_(), raw=True)


# We obtain normalized values by applying the following z-transform:
# 
# \begin{eqnarray}
# z_i=&\frac{x_i - \bar{x}}{\sigma}\\
# \text{with:}&\\
# \bar{x}=&\text{mean}\\
# \sigma=&\text{standarddeviation}
# \end{eqnarray}
# 
# 

# In[31]:


passenger_age['normalized age'] = (passenger_age['Age'] - passenger_age['Age'].mean())/passenger_age['Age'].std()
display(passenger_age)


# Now, let's do the same for another variable: Fare - the price payed for the passage

# In[32]:


passenger_age = passenger_age.merge(train['Fare'] * 10, left_index=True, right_index=True)
passenger_age['normalized_Fare'] = (passenger_age['Fare'] - passenger_age['Fare'].mean())/passenger_age['Fare'].std()
display(passenger_age)


# The un-standardized and standardized variables accross passengers look like this:

# In[38]:


figure, axes = plt.subplots(1, 2, figsize=(15,7))
passenger_age[['Age', 'Fare']].plot(ax=axes[0])
passenger_age[['normalized age', 'normalized_Fare']].plot(ax=axes[1])


# In[39]:


import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../data/titanic/train.csv")
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Fare'] = train['Fare'].fillna(train['Fare'].mean())
fig, ax =plt.subplots(1,2, figsize=(15,7))

sns.distplot(train['Age'],  kde=False, label='Age', ax=ax[0])
sns.distplot(train['Fare'], kde=False, label='Fare', ax=ax[0])
ax[0].legend(prop={'size': 12})
ax[0].title.set_text('Before Normalization')


train['Age'] =(train['Age'] - train['Age'].mean())/train['Age'].std()
train['Fare'] =(train['Fare'] - train['Fare'].mean())/train['Fare'].std()
sns.distplot(train['Age'],  kde=False, label='Age', ax=ax[1])
sns.distplot(train['Fare'], kde=False, label='Fare', ax=ax[1])
ax[1].legend(prop={'size': 12})
ax[1].title.set_text('After Normalization')

fig.show()


# ## not covered here
# The following topics are more advanced and do not apply to tree-methods. In a possible follow-up we can discuss them as well:
#   - power-transforms
#   - mean-encoding
