#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[4]:


import scipy.io


# In[5]:


import matplotlib


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


from sklearn.preprocessing import StandardScaler


# In[8]:


from sklearn.decomposition import PCA


# In[9]:


import sklearn.decomposition as skdc


# In[10]:


from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


import seaborn as sns


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


# Taken from: https://www.kaggle.com/stevebroll/logistic-regression-model-using-pca-components
def standardization(x): #Define function to standardize the data, since all variables are not in the same units
    xmean = np.mean(x) ##calculate mean
    sd = np.std(x) ##calculate standard deviation 
    x_z = (x - xmean) / sd ##calculate standardized value to return
    return(x_z)


# In[15]:


# Step 1: Load in the most recent year of data into a data frame.
#df = pd.read_csv("2015_data.csv", low_memory = False)
df = pd.read_csv("2015_data.csv", nrows = 20000, low_memory = False)


# In[ ]:





# In[16]:


#df.head(100)


# 
# 

# In[17]:


# Analyze the data to determine what can be cleaned, to understand the types of values/inputs. 
for col in df:
    print(col)
    print(df[col].unique())


# In[18]:


# Get number of data records and features.
df.shape


# In[19]:


# Determine how many fields with actual data (not NA fields).
#df.count()


# In[20]:


'''
Lots of missing data:

- education_1989_revision - lots without data
- age_substitution_flag - Lots without data
- infant_age_recode_22 - lots without data
- activity_code - many without data
- place_of_injury_for_causes_w00_y34_except_y06_and_y07_ - many without data
- 130_infant_cause_recode - many without data
- remove entity_condition_19-entity_condition_20
- remove record_condition_3-20
- bridged_race_flag, race_imputation_flag can be removed 

'''

'''
A few nan values:
- manner_of_death
- education_2003_revision
'''

cols_to_remove = ["education_1989_revision", "age_substitution_flag", "infant_age_recode_22", "activity_code", "place_of_injury_for_causes_w00_y34_except_y06_and_y07_", "130_infant_cause_recode", "entity_condition_19", "entity_condition_20", "record_condition_3", "record_condition_4", "record_condition_5", "record_condition_6", "record_condition_7", "record_condition_8", "record_condition_9", "record_condition_10", "record_condition_11", "record_condition_12", "record_condition_13", "record_condition_14", "record_condition_15", "record_condition_16", "record_condition_17", "record_condition_18", "record_condition_19", "record_condition_20", "bridged_race_flag", "race_imputation_flag"]

# Remove the fields that don't contribute enough data (nan fields).
cols_to_keep = []
for col in df:
    if col not in cols_to_remove:
        cols_to_keep.append(col)
        
print(cols_to_keep)        


# In[21]:


# Create the new data frame with only the features we care about/have enough data.
df = df[cols_to_keep]


# In[22]:


df.shape


# In[23]:


#df.count()


# In[24]:


'''
Need to weed out features:
- entity_condition_3-18
'''
cols_to_remove = ["entity_condition_3", "entity_condition_4", "entity_condition_5", "entity_condition_6", "entity_condition_7", "entity_condition_8", "entity_condition_9", "entity_condition_10", "entity_condition_11", "entity_condition_12", "entity_condition_13", "entity_condition_14", "entity_condition_15", "entity_condition_16", "entity_condition_17", "entity_condition_18"]

# Remove the fields that don't contribute enough data (nan fields).
cols_to_keep = []
for col in df:
    if col not in cols_to_remove:
        cols_to_keep.append(col)
        
print(cols_to_keep)        


# In[25]:


# Update the data frame.
df = df[cols_to_keep]


# In[26]:


df.shape


# In[27]:


#df.count()


# In[28]:


# Since classification problem is tied to manner of death, do not want to include any rows without this information.
#df = df[pd.notnull(df['manner_of_death'])]


# In[29]:


df.shape


# In[30]:


# See how this manipulation of the number of records has affected counts. More to weed out?
#df.count()


# In[31]:


'''
Weed out entity_condition_2, record_condition_2
'''

cols_to_remove = ["entity_condition_2", "record_condition_2"]

# Remove the fields that don't contribute enough data (nan fields).
cols_to_keep = []
for col in df:
    if col not in cols_to_remove:
        cols_to_keep.append(col)


# In[32]:


df = df[cols_to_keep]
df.shape


# In[33]:


#df.count()

# Now see that the only field field with nan value is: education_2003_revision.


# In[34]:


# Since education_2003_revision only has values 1-9, we can change the nan fields to contain 0's instead.
df["education_2003_revision"].fillna(0, inplace=True)
df['manner_of_death'].fillna(0, inplace=True)


# In[35]:


# Compute pairwise correlation of all features.
correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title("Correlation between different features")


# In[36]:


# We can drop the highly coorelated values. (> .7)
df = df.drop(['education_reporting_flag'], axis = 1)
df = df.drop(['age_recode_12'], axis = 1)
df = df.drop(['age_recode_27'], axis = 1)
df = df.drop(['age_recode_52'], axis = 1)
df = df.drop(['39_cause_recode'], axis = 1)
df = df.drop(['358_cause_recode'], axis = 1)
df = df.drop(['race_recode_5'], axis = 1)


# In[ ]:





# In[37]:


# In order to use PCA, data cannot contain string (needs numerical data).
# Solution: Use LabelEncoder on data that is ordinal. Use OneHotEncoder on data that is NOT ordinal.
# The following fields are non-numerical: sex, autopsy, marital_status, injury_at_work, method_of_disposition, icd_code_10th_revision,
#                                         entity_condition_1, record_condition_1
# Since non of these are ordinal, we will one hot encode all of them
cols_to_one_hot_encode = ["autopsy", "method_of_disposition", "sex","marital_status", "injury_at_work", "icd_code_10th_revision", "entity_condition_1", "record_condition_1"]

for col in cols_to_one_hot_encode:
    print(col)
    df[col] = pd.Categorical(df[col])
    df_dummies = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df, df_dummies], axis = 1)
    df = df.drop([col], axis = 1)
df.info()


# In[38]:


'''
Feature engineering
Want to use one-hot encoding on non-linear features because don't want to imply a relationship that isn't there is there.

99 - not stated education_1989_revision (replace all 99's with nan, replace nan's with avg)
9 - unknown education_2003_revision
education_reporting_flag - tells which of above items on certificate of death
need all the age_recodes? Or just pick one?
Just pick one of these or chop the one's with it unknown?
Use PCA first, then check correlations. Then can decide what to drop.


age_substitution_flag - calculated age substitutions


place of death - first 3 are same?, where is 8?, 9-unknown - one hot encode, look at PCA output


current_data_year - not relevant (all 2015) but will be when mix in other datasets

manner of death - one hot encode

method_of_disp - one hot

icd_code_10th_revision - unsure what it is, may use in PCA but might scrap later

hot encode cause_recodes (scrap 2, keep 1?) - PCA 

number_of_entity_axis_conditions, entity_condition_1, number_of_record_axis_conditions?, record_condition_1?


race - hot encode

race_recodes - PCA 

hispanic_origin - hot encode (scrap hispanic_originrace_recode)

'''


# In[39]:


#pca.explained_variance_ratio_


# In[40]:


df.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


for col in df:
    print(col)


# In[42]:


# Taken from: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

# Separate out the target (sex) for classification.
# Hack because 2 values for both sex_M and sex_F so both sets of data present the sex with the encoding. Can just pick 1.
y = df.loc[:, ['manner_of_death']].values

df = df.drop(['manner_of_death'], axis = 1)

# Separate out the features.
features =  list(df.columns.values)
x = df.loc[:, features].values

# test_size: what proportion of original data is used for test set
train, test, train_label, test_label = train_test_split( x, y, test_size=1/7.0, random_state=0)

scaler = StandardScaler()
# Fit on training set only.

scaler.fit(train)
# Apply transform to both the training set and the test set.
train = scaler.transform(train)
test = scaler.transform(test)


# In[43]:


# Make an instance of the Model
pca = PCA(.95)


# In[ ]:





# In[44]:


pca.fit(train)


# In[45]:


train = pca.transform(train)
test = pca.transform(test)


# In[46]:


# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')


# In[ ]:





# In[47]:


#logisticRegr.score(test, test_label)


# In[48]:


'''# Standardize the data before performing PCA. Needs to be on unit scale for optimal performance of algorithms.
# Separate out the features.
features =  list(df.columns.values)
x = df.loc[:, features].values

# Separate out the target (sex) for classification.
# Hack because 2 values for both sex_M and sex_F so both sets of data present the sex with the encoding. Can just pick 1.
y = df.loc[:, ['sex_M']].values

x = StandardScaler().fit_transform(x)

pca = PCA(n_components = 2)

principal_components = pca.fit_transform(x)

principal_df = pd.DataFrame(data = principal_components, columns = ['principal_component_1', 'principal_component_2'])

result_df = pd.concat([principal_df, df[['sex_M']]], axis = 1)

# Plot the result.
fig = plt.figure(figsize = (8,8))
axis = fig.add_subplot(1,1,1)
axis.set_xlabel('Principal Component 1', fontsize = 15)
axis.set_ylabel('Principal Component 2', fontsize = 15)
axis.set_title('2 Component PCA', fontsize = 20)

targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indices_to_keep = result_df['sex_M'] == target
    axis.scatter(result_df.loc[indices_to_keep, 'principal_component_1'], result_df.loc[indices_to_keep, 'principal_component_2'], c = color, s = 50)
    
axis.legend(targets)
axis.grid()
'''


# In[49]:


logisticRegr.fit(train, train_label)


# In[50]:


logisticRegr.score(train, train_label)


# In[51]:


print('Coefficient: \n', logisticRegr.coef_)
print('Intercept: \n', logisticRegr.intercept_)


# In[52]:


test.shape


# In[53]:


#logisticRegr.predict(test)
predicted = logisticRegr.predict(test[0].reshape(1,-1))


# In[54]:


print(predicted)


# In[55]:


logisticRegr.score(test, test_label)


# In[ ]:




