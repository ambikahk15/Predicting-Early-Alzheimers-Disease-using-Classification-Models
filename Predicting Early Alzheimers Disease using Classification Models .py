#!/usr/bin/env python
# coding: utf-8

# ## Importing And Loading Dataset

# In[189]:


# Importing all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[190]:


# reading dataset
alzheimer_dataset = pd.read_csv("oasis_longitudinal.csv", sep = ",")


# In[191]:


alzheimer_dataset.head(10)


# In[192]:


# size of the dataset
alzheimer_dataset.shape


# ## Exploring the Dataset

# In[193]:


alzheimer_dataset.info()


# In[194]:


# number of nulls in each column
alzheimer_dataset.isnull().sum()


# In[195]:


# Number of Numercial and categorrical columns
numerical_col=alzheimer_dataset.select_dtypes(include=np.number).columns
print("numerical columns: \n\n",numerical_col)
print('\n')
categorical_col=alzheimer_dataset.select_dtypes(exclude=np.number).columns
print("categorical columns: \n\n",categorical_col)


# In[196]:


# description of the dataset
alzheimer_dataset.describe()


# ## Exploratory Data Analysis (with Converted Group)

# In[197]:


sns.countplot(alzheimer_dataset['Group']).set_title("Barplot for Alzheimer status")


# In[198]:


sns.countplot(x = 'Group', hue = 'Visit', data = alzheimer_dataset).set_title("Barplot for Alzheimer status")


# In[199]:


sns.countplot(x = 'Visit', hue = 'Group', data = alzheimer_dataset).set_title("Barplot for Alzheimer status")


# In[200]:


sns.distplot(alzheimer_dataset['EDUC'], color='Purple')
plt.title('Distribution of Education among patients', fontsize=15)
plt.xlabel('Years', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.show()


# In[201]:


sns.catplot(x='Group', y='Age', hue = 'M/F', data=alzheimer_dataset, kind='box')


# In[202]:


sns.swarmplot(x="Group", y="MR Delay", data = alzheimer_dataset)


# In[203]:


sns.countplot(x="CDR", hue="Group", data = alzheimer_dataset)
plt.title('Dementia with Clinical Dementia rating', fontsize=15)
plt.show()


# In[204]:


sns.countplot(x = 'SES', hue = 'Visit', data = alzheimer_dataset).set_title("Barplot for Alzheimer status")


# ## Data Preprocessing

# In[205]:


# Considering only Visit = 1 patients 
alzheimer_dataset = alzheimer_dataset[alzheimer_dataset['Visit'] == 1]


# In[206]:


# Checking for Null Values
alzheimer_dataset.isnull().sum()


# In[207]:


# Dropping the missing value rows [with dropping columns]
alzheimer_dataset_drp =  alzheimer_dataset.dropna()


# In[208]:


alzheimer_dataset_drp.shape


# In[209]:


#Replacing null with Mean of the columns [With Imputation]
mean_value = alzheimer_dataset['SES'].mean()
alzheimer_dataset['SES'].fillna(value = mean_value, inplace = True)


# In[210]:


alzheimer_dataset.shape


# In[211]:


alzheimer_dataset


# In[212]:


# Considering "Converted" Observation as "Demented" 
alzheimer_dataset['Group'] = alzheimer_dataset['Group'].replace(['Converted'], ['Demented'])
# Considering "Converted" Observation as "Demented"
alzheimer_dataset_drp['Group'] = alzheimer_dataset_drp['Group'].replace(['Converted'], ['Demented'])


# In[213]:


# Converting all Categorical values for dataset with imputation
from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()

alzheimer_dataset['M/F'] = lab.fit_transform(alzheimer_dataset['M/F'])
alzheimer_dataset['Group'] = lab.fit_transform(alzheimer_dataset['Group'])
alzheimer_dataset['Hand'] = lab.fit_transform(alzheimer_dataset['Hand'])


# In[214]:


# Converting all Categorical values for dataset with dropping columns
from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()

alzheimer_dataset_drp['M/F'] = lab.fit_transform(alzheimer_dataset_drp['M/F'])
alzheimer_dataset_drp['Group'] = lab.fit_transform(alzheimer_dataset_drp['Group'])
alzheimer_dataset_drp['Hand'] = lab.fit_transform(alzheimer_dataset_drp['Hand'])


# In[215]:


alzheimer_dataset


# In[216]:


alzheimer_dataset_drp


# In[217]:


# for imputed dataset
alzheimer_dataset.groupby('Group').size()


# In[218]:


# for observations dropped dataset
alzheimer_dataset_drp.groupby('Group').size() 


# ## Exploratoty Data Analysis (without Converted group)

# In[219]:


sns.countplot(alzheimer_dataset['Group']).set_title("Barplot for Alzheimer status")


# In[220]:


sns.boxplot(x="Group", y="nWBV", data = alzheimer_dataset)
plt.title('Dementia relation with brain volume', fontsize=15)


# In[221]:


sns.countplot(x="CDR", hue="Group", data = alzheimer_dataset)
plt.title('Dementia with Clinical Dementia rating', fontsize=15)
plt.show()


# In[222]:


sns.barplot(x="Group", y="MMSE", data = alzheimer_dataset)
plt.title('Dementia with mini mental examination Score', fontsize=15)
plt.show()


# In[223]:


sns.countplot( x = 'EDUC', hue = 'Group', data = alzheimer_dataset)
plt.title('Distribution of Education among  patients', fontsize=15)
plt.show()


# In[224]:


sns.catplot(x='Group', y='Age', hue = 'M/F', data = alzheimer_dataset, kind='box')
plt.title("Box Plot of Age by Group(demented/nondemented), Separated by Sex")


# In[225]:


new_df = alzheimer_dataset.loc[alzheimer_dataset['Group'] == 1]
new_df_1 = alzheimer_dataset.loc[alzheimer_dataset['Group'] == 0]
sns.distplot(new_df['EDUC'], color='Purple')
sns.distplot(new_df_1['EDUC'], color='Green')
plt.title('Distribution of Education among Demented patients', fontsize=15)
plt.xlabel('Years', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.show()


# ## Further Analysis

# In[226]:


# Droping not required columns from imputed dataset
alzheimer_dataset =  alzheimer_dataset.drop(["Visit", "MR Delay", "Subject ID", "MRI ID", "Hand"], axis = 1)


# In[227]:


# Droping not required columns from drop dataset
alzheimer_dataset_drp =  alzheimer_dataset_drp.drop(["Visit", "MR Delay", "Subject ID", "MRI ID", "Hand"], axis = 1)


# In[228]:


alzheimer_dataset.corr()


# In[229]:


alzheimer_dataset_drp.corr()


# In[230]:


sns.heatmap(alzheimer_dataset.corr(), annot = True)


# In[231]:


sns.heatmap(alzheimer_dataset_drp.corr(), annot = True)


# There are two features correlated to group status of the observation.
# 1) MMSE = 0.53 means MMSe is directly proportional to group status
# 2) CDR = -0.77 means CDR is inversely proportional to group status

# # Splitting Dataset

# In[232]:


from sklearn.model_selection import train_test_split


# In[233]:


features = ['M/F', 'Age', 'EDUC', 'MMSE', 'SES', 'CDR', 'eTIV', 'nWBV', 'ASF']


# In[234]:


X = alzheimer_dataset[features]  # Features
y = alzheimer_dataset.Group  # target


# In[235]:


# Data-split for Imputed dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)


# In[236]:


X1 = alzheimer_dataset_drp[features]  # Features
y1= alzheimer_dataset_drp.Group  # target


# In[237]:


# Data - split for dropped dataset
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1, test_size = 0.25, random_state = 0)


# # Alzheimers Classification Based on Logistic Regression

# ### Logistic Regression on  Imputed dataset

# In[238]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV 


# In[239]:


lr = LogisticRegression()


# In[240]:


parameters = {'C':[0.0001, 0.001, 0.01, 0.1,1,0.5,0.05,0.005,5,50,10,100,1000]}


# In[241]:


grid_lr = GridSearchCV(lr, parameters, cv = 5, return_train_score = False)


# In[242]:


grid_lr.fit(X_train, y_train)


# In[243]:


print(grid_lr.best_params_)


# In[244]:


df = pd.DataFrame(grid_lr.cv_results_)


# In[245]:


df[['param_C', 'mean_test_score']]


# In[246]:


best_score = grid_lr.best_score_
print(best_score)


# In[247]:


lr = LogisticRegression(C = 10)


# In[248]:


#fit the model with the training data
lr.fit(X_train, y_train)

# Make a prediction for the testing data

y_pred_lr = lr.predict(X_test)


# In[249]:


cnf_matrix = metrics.confusion_matrix(y_test, y_pred_lr)
print(cnf_matrix)


# In[250]:


print(classification_report(y_test, y_pred_lr))


# In[251]:


print("Best accuracy on validation set is:", best_score)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_lr))
print("Precision:", metrics.precision_score(y_test, y_pred_lr, average = 'weighted'))
print("Recall:", metrics.recall_score(y_test, y_pred_lr, average = 'weighted'))
print("F1-Score:", metrics.f1_score(y_test, y_pred_lr, average = 'weighted'))


# ### Logistic Regression on observations dropped dataset

# In[252]:


lr = LogisticRegression()


# In[253]:


parameters = {'C':[0.0001, 0.001, 0.01, 0.1,1,0.5,0.05,0.005,5,50]}


# In[254]:


grid_lr = GridSearchCV(lr, parameters, cv = 5, return_train_score = False)


# In[255]:


grid_lr.fit(X1_train, y1_train)


# In[256]:


print(grid_lr.best_params_)


# In[257]:


df = pd.DataFrame(grid_lr.cv_results_)


# In[258]:


df[['param_C', 'mean_test_score']]


# In[259]:


best_score = grid_lr.best_score_
print(best_score)


# In[260]:


lr = grid_lr.best_estimator_


# In[261]:


# Make a prediction for the testing data

y1_pred_lr = lr.predict(X1_test)


# In[262]:


cnf_matrix = metrics.confusion_matrix(y1_test, y1_pred_lr)
print(cnf_matrix)


# In[263]:


print(classification_report(y1_test, y1_pred_lr))


# In[264]:


print("Best accuracy on validation set is:", best_score)
print("Accuracy:", metrics.accuracy_score(y1_test, y1_pred_lr))
print("Precision:", metrics.precision_score(y1_test, y1_pred_lr, average = 'weighted'))
print("Recall:", metrics.recall_score(y1_test, y1_pred_lr, average = 'weighted'))
print("F1-Score:", metrics.f1_score(y1_test, y1_pred_lr, average = 'weighted'))


# # Alzheimers Classification Based on RandomForest

# ### RandomForest Classifier on Imputed dataset

# In[265]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, recall_score, precision_score, f1_score


# In[158]:


best_score = 0


# In[159]:


for M in range(2, 15, 2): # combines M trees
    for d in range(1, 9): # maximum number of features considered at each split
        for m in range(1, 9): # maximum depth of the tree

            forestModel = RandomForestClassifier(n_estimators=M, max_features=d, n_jobs=4,
                                          max_depth=m, random_state=0)

            scores = cross_val_score(forestModel, X_train, y_train, cv=5, scoring='accuracy')

            score = np.mean(scores)

            if score > best_score:
                best_score = score
                best_M = M
                best_d = d
                best_m = m


# In[160]:


SelectedRFModel = RandomForestClassifier(n_estimators=M, max_features=d,
                                          max_depth=m, random_state=0).fit(X_train, y_train )


# In[161]:


PredictedOutput = SelectedRFModel.predict(X_test)


# In[162]:


test_score = SelectedRFModel.score(X_test, y_test)
test_recall = recall_score(y_test, PredictedOutput, pos_label=1)
test_precision = precision_score(y_test, PredictedOutput, pos_label =1)
f1_score = f1_score(y_test, PredictedOutput, pos_label =1)


# In[163]:


print("Best accuracy on validation set is:", best_score)
print("Best parameters of M, d, m are: ", best_M, best_d, best_m)
print("Test accuracy with the best parameters is", test_score)
print("Test recall with the best parameters is:", test_recall)
print("Test precision with the best parameters is", test_precision)
print("Test F1 score with the best parameters is:", f1_score)


# ### RandomForest classifier on Dropped observation dataset

# In[266]:


best_score = 0


# In[267]:


for M in range(2, 15, 2): # combines M trees
    for d in range(1, 9): # maximum number of features considered at each split
        for m in range(1, 9): # maximum depth of the tree

            forestModel = RandomForestClassifier(n_estimators=M, max_features=d, n_jobs=4,
                                          max_depth=m, random_state=0)

            scores = cross_val_score(forestModel, X1_train, y1_train, cv=5, scoring='accuracy')

            score = np.mean(scores)

            if score > best_score:
                best_score = score
                best_M = M
                best_d = d
                best_m = m


# In[268]:


SelectedRFModel = RandomForestClassifier(n_estimators=M, max_features=d,
                                          max_depth=M, random_state=0).fit(X1_train, y1_train )


# In[269]:


PredictedOutput = SelectedRFModel.predict(X1_test)


# In[270]:


test_score = SelectedRFModel.score(X1_test, y1_test)
test_recall = recall_score(y1_test, PredictedOutput, pos_label=1)
test_precision = precision_score(y1_test, PredictedOutput, pos_label =1)
f1_score = f1_score(y1_test, PredictedOutput, pos_label =1)


# In[271]:


print("Best accuracy on validation set is:", best_score)
print("Best parameters of M, d, m are: ", best_M, best_d, best_m)
print("Test accuracy with the best parameters is", test_score)
print("Test recall with the best parameters is:", test_recall)
print("Test precision with the best parameters is", test_precision)
print("Test F1 score with the best parameters is:", f1_score)


# In[ ]:





# # Alzheimers Classification Based on SVM

# ### SVM on Imputed dataset

# In[272]:


from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC


# In[273]:


parameters = {'kernel':['linear', 'poly'],
              'C':[0.1, 1, 100],
              'gamma':[0.001, 0.01, 0.1, 1]}


# In[274]:


model = SVC( )


# In[87]:


from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(model, parameters, cv = 5, return_train_score = False, n_iter = 5 )


# In[ ]:


pd.DataFrame(rs.cv_results_)[['param_C', 'param_gamma','mean_test_score']]


# In[278]:


print(rs.best_params_, rs.best_score_)


# In[ ]:


model_best = rs.best_estimator_ 


# In[283]:


model_best.fit(X_train, y_train)


# In[ ]:


print(rs.best_params_, rs.best_score_)


# In[284]:


predicted = model_best.predict(X_test)


# In[285]:


cnf_matrix = metrics.confusion_matrix(y_test, predicted)
print(cnf_matrix)


# In[286]:


print(classification_report(y_test, predicted))


# In[287]:


print("Accuracy:", metrics.accuracy_score(y_test, predicted))
print("Precision:", metrics.precision_score(y_test, predicted, average = 'weighted'))
print("Recall:", metrics.recall_score(y_test, predicted, average = 'weighted'))
print("F1-Score:", metrics.f1_score(y_test, predicted, average = 'weighted'))


# In[ ]:





# ### SVM on observation Dropped Dataset

# In[288]:


model_best.fit(X1_train, y1_train)


# In[ ]:


pd.DataFrame(rs.cv_results_)[['param_C', 'param_gamma','mean_test_score']]


# In[ ]:


print(rs.best_params_, rs.best_score_)


# In[248]:


model_best = rs.best_estimator_


# In[289]:


predicted = model_best.predict(X1_test)


# In[290]:


cnf_matrix = metrics.confusion_matrix(y1_test, predicted)
print(cnf_matrix)


# In[291]:


print(classification_report(y1_test, predicted))


# In[292]:


print("Accuracy:", metrics.accuracy_score(y1_test, predicted))
print("Precision:", metrics.precision_score(y1_test, predicted, average = 'weighted'))
print("Recall:", metrics.recall_score(y1_test, predicted, average = 'weighted'))
print("F1-Score:", metrics.f1_score(y1_test, predicted, average = 'weighted'))


# In[ ]:




