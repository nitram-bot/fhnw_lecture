#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')


# 
# 1. Morning
#     - __Data Modelling & Cross Validation__
#     - data leakage & dependent data
#     - imbalanced data (example in python)
#     - study: Ebanking Fraud 
#     - Q&A
#     
#     <br>
#     
#     
# 2. Afternoon
#     - Data Basics & historical perspective
#     - Linear Regression
#     - Trees
#     - house prices (regression example in python)
#     - Clustering
#     - bonus: Hyperparameter Optimization and AutoML
#     - Q&A

# # Data Modelling for data with temporal context
# ![](../images/banking_recommender1.png)

# ![](../images/banking_recommender2.png)

# # Overfitting and Cross-Validation
# All classification and regression algorithms are prone to overfitting:<br>
# The algorithms learn pecularities of the train-data, that are not present in the real-world data.<br> 
# When over-fitted, the algorithms are not generalizing to the real data.<br>
# 
# __Capacity__ refers to the ratio of free parameters and the amount of training data.

# In[3]:


Image('../images/bias_variance_tradeoff.png')


# # Cross-Validation
# In most real-word applications we do not know the data universe, i.e. we do not know all possible data points that might be there. Our training data is possibly just a biased subsample of the population.<br>
# When we fit our algorithm to such a subsample its performance will degrade, when applied to new, unseen data points. In order to have an idea, how well our algorithm will perform in such cases, we can use a cross-validation scheme:<br>
# In the example below, a 5-fold cross-validation is illustrated.
# * split the training data in 5 equal sized parts. In *sklearn* you can choose *StratifiedKFold*, that essentially tries to keep the percentages of all classes stable within each fold.
# * train your algorithmm on 4 folds and classify data in the 5th hold-out fold. Keep the performance on this fold.
# * repeat the last step 4 more times and use each time another fold as your hold-out fold.
# * at the end, you have 5 independent estimates of your algorithm's performance
# * compute the mean of theses 5 estimates for an overall estimate

# In[4]:


Image('../images/cross_validation.png')


# 
# 1. Morning
#     - Data Modelling & Cross Validation
#     - __data leakage & dependent data__
#     - imbalanced data (example in python)
#     - study: Ebanking Fraud 
#     - Q&A
#     
#     <br>
#     
#     
# 2. Afternoon
#     - Data Basics & historical perspective
#     - Linear Regression
#     - Trees
#     - house prices (regression example in python)
#     - Clustering
#     - bonus: Hyperparameter Optimization and AutoML
#     - Q&A

# # Date Leakage and Dependent Data

# In[8]:


from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')


# In[9]:


Image("../images/data-leakage explanation.png")
# <img alt="" caption="how data leakage might happen" 
# id="data_leakage" src="../images/data-leakage explanation.png" width="640" height="640">
# image taken form [here](https://towardsdatascience.com/how-data-leakage-affects-machine-learning-models-in-practice-f448be6080d0)


# # Sources of data leakage
# ### train data contains features that are not available in production
# e.g., the row-number contains information about the target: first come the negative examples, the positive cases were then simply inserted underneath.  

# ### future data somehow slipped into the training set
# e.g. Giba's property:
# [taken from kaggle](https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/61329)<br>

# In[10]:


Image("../images/Giba_santander.png")
# <img alt="" caption="how data leakage might happen" 
# id="data_leakage" src="../images/Giba_santander.png" width="640" height="640"><br>


# and here is the mentioned data-structure:<br>
# [this kernel exploits the leakage](https://www.kaggle.com/rebeccaysteboe/giba-s-property-extended-result)

# In[11]:


Image("../images/Gibas_data_structure.png") 
# <img alt="" caption="how data leakage might happen" 
# id="data_leakage" src="../images/Gibas_data_structure.png" width="640" height="640"><br>


# ### there is one feature that interacts with the target
# taken from [Breast Cancer Identification: KDD CUP Winner’s Report](http://kdd.org/exploration_files/KDDCup08-P1.pdf)<br>
# Distribution of malignant (black) and benign (gray) candidates depending on patient ID on the X-axis in log scale.

# In[12]:


Image("../images/Distribution-of-malignant-black-and-benign-gray-candidates-depending-on-patient-ID-on.png")
# <img alt="" caption="how data leakage might happen" 
# id="data_leakage" src="../images/Distribution-of-malignant-black-and-benign-gray-candidates-depending-on-patient-ID-on.png" width="640" height="640"><br>


# ### Some more cases where we have data leakage: 
#    - Customer advisor has a long call with customer and finally sells the product that is shipped only two weeks later. Variables 'last advisory contact' and 'length of call' certainly anticipate the product sale. When an algorithm learns to predict product propensity based on 'last advisor contact' it will ultimately suggest customers to the advisors for whom the advisor has already closed the deal.
#    - Train and test data is normalized with common sample statistics belonging to the whole data set
#         * target encoding is dangerous: we will talk about it later on
#         * stacking is dangerous: we will discuss this topic as well

# ### Example: credit card applications
# This data example was used in "Econometric Analysis" (William H. Greene) without the author noticing the problem:<br>
# [example taken from here](https://www.kaggle.com/dansbecker/data-leakage)
#  - card: Dummy variable, 1 if application for credit card accepted, 0 if not
#  - reports: Number of major derogatory reports
#  - age: Age n years plus twelfths of a year
#  - income: Yearly income (divided by 10,000)
#  - share: Ratio of monthly credit card expenditure to yearly income
#  - expenditure: Average monthly credit card expenditure
#  - owner: 1 if owns their home, 0 if rent
#  - selfempl: 1 if self employed, 0 if not.
#  - dependents: 1 + number of dependents
#  - months: Months living at current address
#  - majorcards: Number of major credit cards held
#  - active: Number of active credit accounts
# 

# In[28]:


import pandas as pd

url = 'https://raw.githubusercontent.com/YoshiKitaguchi/Credit-card-verification-project/master/AER_credit_card_data.csv'
df = pd.read_csv(url, error_bad_lines=False, true_values = ['yes'], false_values = ['no'])
print(df.head())


# In[14]:


get_ipython().system('pip install lightgbm imblearn')


# In[29]:


from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np
import lightgbm

y = df['card']
X = df.drop('card', axis=1)


# In[16]:


model = lightgbm.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1, 
                                n_estimators=500, subsample_for_bin=20000, objective='binary', 
                                subsample=1.0, subsample_freq=0, colsample_bytree=1.0, 
                                n_jobs=- 1, silent=True, importance_type='split',
                                is_unbalance = False, scale_pos_weight = 1.0)
model_pipe = make_pipeline(model)
cv_scores = cross_val_score(model_pipe, X, y, scoring='accuracy')
print(np.mean(cv_scores))


# In[20]:


from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
model.fit(X, y)
result = permutation_importance(model, X, y,
        n_repeats=30,
        random_state=0)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=X.columns[sorted_idx])
ax.set_title("Permutation Importances")
fig.tight_layout()
plt.show()


# In[10]:


import matplotlib.pyplot as plt 
warnings.filterwarnings('ignore')
fig,axes =plt.subplots(5,2, figsize=(12, 9)) # 3 columns each containing 10 figures, total 30 features
# malignant=cancer.data[cancer.target==0] # define malignant
# benign=cancer.data[cancer.target==1] # define benign
ax=axes.ravel()# flat axes with numpy ravel
for i, col in enumerate(X.columns.tolist()[0:10]):
  _,bins=np.histogram(X[col])
  ax[i].hist(X.loc[y == True, col],bins=bins,color='r',alpha=.5)
  ax[i].hist(X.loc[y == False, col],bins=bins,color='g',alpha=0.3)
  ax[i].set_title(col, fontsize=9)
  ax[i].axes.get_xaxis().set_visible(True) # the x-axis co-ordinates are not so useful, as we just want to look how well separated the histograms are
  ax[i].set_yticks(())
ax[0].legend(['True','False'],loc='best',fontsize=8)    
plt.tight_layout()# let's make good plots
plt.show() 


# In[11]:


display(X.loc[y == True, 'expenditure'].mean(), X.loc[y == False, 'expenditure'].mean())


# In[12]:


display(X.loc[y == True, 'share'].mean(), X.loc[y == False, 'share'].mean())


# In[13]:


get_ipython().system('pip install dtreeviz')


# In[3]:


from sklearn import tree
from dtreeviz.trees import *
import matplotlib.pyplot as plt
classifier = tree.DecisionTreeClassifier(max_depth=3)  # limit depth of tree
classifier.fit(X, y)


viz = dtreeviz(classifier, 
               X.values,
               y.values, 
               target_name='credit-card application',
               feature_names=X.columns.tolist(),
               class_names = ['not_accepted', 'accepted']
              )  

viz.save("decision_tree.svg") 


# In[15]:


# from IPython.core.display import HTML
# HTML(''' <' img src='decision_tree.svg' / > ''')
import IPython
#IPython.display.SVG('decision_tree.svg')


# In[3]:


# from IPython.display import SVG, display
# from IPython.display import display, Markdown
# display(Markdown('![figure](decision_tree.svg)'))
# warnings.filterwarnings('ignore')
# viz
# warnings.filterwarnings('once')
Image("decision_tree.png")


# ## Solution:
# Obviously, 
#   - share: Ratio of monthly credit card expenditure to yearly income
#   - expenditure: Average monthly credit card expenditure
# 
# are features that suppose the applicant was granted a credit card.

# # Dependency between data-samples
# 
# Training Machine Learning Algorithms works best, when we have many independent data samples in the training data. Dependent data arises when:
# 
#   - we take repeatedly measures from the same individual (the trained algorithms will not generalize to other individuals)
#   - we take samples only from one bank (the socia-demographic structure of GKB's customers might be differentfrom that of BCG's customers - as a result the algorithm will badly generalize)

# ## Some more cases where we have dependent data
#  - Repeatedly sampling data from the same individual:
#      * Fraud: A fraudster commits many frauds that have a similiar pattern; For example for every fraud commited, the fraudster uses a different account of the same bank in Thailand. When doing cross-validation we have frauds related to the very bank in Thailand in the training set as well as in the test set. Hence, we will overestimate the capability of the trained classifier to generalize to new, unseen fraud cases. But it will be very efficient to detect this one fraudster with bank accounts in Thailand.
#      * customer journey: to detect an event as soon as possible, data is sampled with different offsets before the event's occurence. When trying to predict an event we could be tempted to sample data from different points in time before the event. This data will always be very similar and is hence dependent. For example, medical health records are not changing very fast and blood pressure two months ago will be similar to that measured one month ago. Most bank accounts have a similar balance in a one months distance.
#      * classifying websites: social media websites belong all to facebook. There are just not enough social media websites to learn something about them in the training set and generalize to other social media websites in the test set. 
#  - Train and test data is normalized with common sample statistics belonging to the whole data set
#      * target encoding is dangerous: we will talk about it later on
#      * stacking is dangerous: we will discuss this topic as well

#      
#  - Sentence Classification: sentences belonging to the same document
#      * customer churn: An angry customer sends frequent e-mails. All e-mails happen to have the same characteristics, e.g. instead of the *Umlaut* 'ü', the customer uses 'ue'. The algorithm might be tempted to learn that 'ue' is a special churn-characteristic. When half of the e-mails end up in the train set and the rest in the test set, we will overestimate the prediction accuracy of the learned algorithm.
#      * When building a classifier to distinct medical publications from IT-related publications, it is important to have a representative sample of medical topics as well as tech-topics. When taking sentences from one document that is heavily Java related, the algorithm will struggle to generalize to the programming language Python. When the 'Java-sentences' are in the train set as well as in the test set, we will overestimate the algorithm's performance on new documents.
#  - Diagnosis: patient records coming from the same hospital
#      * Hospitals might have different specializations; When we want to predict diagnosis based on the doctors' reports, cancer cases from a clinic specialized in cancer treatments might have higher similarity to each other than cancer cases coming from a orthopaedic hospital. When the reports of the specialized clinic end up in the train set as well as in the test set, we will overestimate the algorithm's capability to correctly classifiy the diagnosis 'cancer'. 
#        

# ### recent article summarizing errors when predicting COVID:
# Excerpts of the article [Hundreds of AI tools have been built to catch covid. None of them helped.](https://www.technologyreview.com/2021/07/30/1030329/machine-learning-ai-failed-covid-hospital-diagnosis-pandemic):<br>
# "They looked at 415 published tools and, like Wynants and her colleagues, concluded that none were fit for clinical use."<br>
# "Both teams found that researchers repeated the same basic errors in the way they trained or tested their tools."<br>
# 
# "Many of the problems that were uncovered are linked to the poor quality of the data that researchers used to develop their tools."<br>
# 
#  - __duplicates:__ "Driggs highlights the problem of what he calls Frankenstein data sets, which are spliced together from multiple sources and can contain duplicates."
#  - __confounding variables:__ <br>
#    * "Many unwittingly used a data set that contained chest scans of children who did not have covid as their examples of what non-covid cases looked like. But as a result, the AIs learned to identify kids, not covid."
#    * "Because patients scanned while lying down were more likely to be seriously ill, the AI learned wrongly to predict serious covid risk from a person’s position."
#  - __different sources:__ "In yet other cases, some AIs were found to be picking up on the text font that certain hospitals used to label the scans. As a result, fonts from hospitals with more serious caseloads became predictors of covid risk."
#  - __human labeling error:__ "It would be much better to label a medical scan with the result of a PCR test rather than one doctor’s opinion, says Driggs."
#  <br>

#  
# ### How to fix it?
# 
#  - "Better data would help, but in times of crisis that’s a big ask."<br>
#  - “'Until we buy into the idea that we need to sort out the unsexy problems before the sexy ones, we’re doomed to repeat the same mistakes,' says Mateen."
# <br>
# 
# Original articles:<br>
# [Wynants et al., 2020. Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal](https://www.bmj.com/content/369/bmj.m1328)<br>
# [Roberts et al., 2021. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans](https://www.nature.com/articles/s42256-021-00307-0)

# ### Data leakage Literature
# 
# * [examples of data leakage in competitions: start on page 19](https://static1.squarespace.com/static/5a4c161cfe54ef45b17aa18e/t/5ab4013b88251b7b684c6025/1521746286132/week2-part2.pdf)<br>
# * [alternative to the text above here is the video](https://www.coursera.org/lecture/competitive-data-science/basic-data-leaks-5w9Gy)<br>
# * [Medical data mining: insights from winning two competitions](https://www.prem-melville.com/publications/medical-mining-dmkd09.pdf)
# 
# __important__: if you're not allowed to read any more 'towardsdatascience' article or 'medium' articles -- just remove the cookies for the page (inspect -> applications -> cookies) and reload the page afterwards.<br>
# * [data leakage when tuning hyper-parameters](https://towardsdatascience.com/data-leakage-with-hyper-parameter-tuning-c57ba2006046)
