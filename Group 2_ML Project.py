#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Prediction from a Healthcare Dataset 

# #### Necessary libraries to be loaded 

# In[397]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[398]:


df = pd.read_csv(r'C:\Users\Souvik Das\Downloads\ML\Project\breast cancer.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# In[399]:


df.head()


# In[400]:


df.shape


# In[401]:


df.describe()


# In[402]:


df.diagnosis.unique()


# In[403]:


df['diagnosis'].value_counts()


# In[404]:


sns.countplot(df['diagnosis'], palette='hls')
plt.savefig('Losgistic.png')


# ### Cleaning the dataset

# In[405]:


df.drop('id',axis=1,inplace=True)


# In[406]:


df.head()


# In[407]:


df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()


# In[408]:


df.isnull().sum()


# In[409]:


df.corr()


# we can see that radius_mean , perimeter _mean, area_mean have a high correlation with malignant tumor

# In[410]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)
plt.savefig('CorrelationMatrix.png')

Multicollinearity is a problem as it undermines the significance of independent varibales and we fix it 
by removing the highly correlated predictors from the model
Use Partial Least Squares Regression (PLS) or Principal Components Analysis, regression methods that cut the number 
of predictors to a smaller set of uncorrelated components.
# In[411]:


# Generate and visualize the correlation matrix
corr = df.corr().round(2)

# Mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set figure size
f, ax = plt.subplots(figsize=(20, 20))

# Define custom colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.tight_layout()
plt.savefig('Heatmap.png')

We can verify the presence of multicollinearity between some of the variables. 
For instance, the radius_mean column has a correlation of 1 and 0.99 with perimeter_mean and area_mean columns, respectively.
This is because the three columns essentially contain the same information, which is the physical size of the observation
(the cell). 
Therefore we should only pick ONE of the three columns when we go into further analysis.Another place where multicollienartiy is apparent is between the "mean" columns and the "worst" column.
For instance, the radius_mean column has a correlation of 0.97 with the radius_worst column.Also there is multicollinearity between the attributes compactness, concavity, and concave points. So we can choose just ONE out of these, I am going for Compactness.
# In[412]:


# first, drop all "worst" columns
cols = ['radius_worst', 
        'texture_worst', 
        'perimeter_worst', 
        'area_worst', 
        'smoothness_worst', 
        'compactness_worst', 
        'concavity_worst',
        'concave points_worst', 
        'symmetry_worst', 
        'fractal_dimension_worst']
df = df.drop(cols, axis=1)

# then, drop all columns related to the "perimeter" and "area" attributes
cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']
df = df.drop(cols, axis=1)

# lastly, drop all columns related to the "concavity" and "concave points" attributes
cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
df = df.drop(cols, axis=1)

# verify remaining columns
df.columns


# In[413]:


# Draw the heatmap again, with the new correlation matrix
corr = df.corr().round(2)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.tight_layout()
plt.savefig('RevisedHeatmap.png')


# # Model Building

# In[414]:


X=df.drop(['diagnosis'],axis=1)
y = df['diagnosis']


# In[415]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=40)


# ### Feature Scaling 

# In[416]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)


# # Finding Out The Best Models

# #### Common libraries to be imported

# In[417]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# ### Logistic Regression 

# In[418]:


from sklearn.linear_model import LogisticRegression


# In[419]:


lr=LogisticRegression()

model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)


# In[420]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
lr_probs = model1.predict_proba(X_test)

# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()
plt.savefig('ROCR.png')


# In[421]:


cm=confusion_matrix(y_test,prediction1)
cm


# In[422]:


sns.heatmap(cm,annot=True,cmap ="YlOrBr",linecolor='White', linewidths=1)
plt.savefig('Losgistic.png')


# In[423]:


accuracy_score(y_test,prediction1)


# In[424]:


print(classification_report(y_test, prediction1))


# ### Decision Tree

# In[425]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt


# In[426]:


dtc=DecisionTreeClassifier()
model2=dtc.fit(X_train,y_train)
prediction2=model2.predict(X_test)


# In[427]:


text_representation = tree.export_text(dtc)
print(text_representation)


# In[428]:


cm2= confusion_matrix(y_test,prediction2)
cm2


# In[429]:


sns.heatmap(cm2,annot=True, cmap ="YlGn",linecolor='White', linewidths=1)
plt.savefig('DecisionTree.png')


# In[430]:


accuracy_score(y_test,prediction2)


# In[431]:


print(classification_report(y_test, prediction2))


# ### Random Forest

# In[432]:


from sklearn.ensemble import RandomForestClassifier


# In[433]:


rfc=RandomForestClassifier()
model3 = rfc.fit(X_train, y_train)
prediction3 = model3.predict(X_test)
cm3=confusion_matrix(y_test, prediction3)
cm3


# In[434]:


sns.heatmap(cm3,annot=True, cmap ="BuPu",linecolor='White', linewidths=1)
plt.savefig('RandomForest.png')


# In[435]:


accuracy_score(y_test, prediction3)


# In[436]:


print(classification_report(y_test, prediction3))


# ### Support Vector Machine

# In[437]:


from sklearn import svm


# In[438]:


model4 = svm.SVC(kernel='rbf',C=30,gamma='auto')
model4.fit(X_train,y_train)
prediction4 = model4.predict(X_test)
cm4=confusion_matrix(y_test, prediction4)
cm4


# In[439]:


sns.heatmap(cm4,annot=True, cmap ="GnBu",linecolor='White', linewidths=1)
plt.savefig('SVM.png')


# In[440]:


accuracy_score(y_test, prediction4)


# In[441]:


print(classification_report(y_test, prediction4))


# ### Naive Bayes

# In[442]:


from sklearn.naive_bayes import GaussianNB


# In[443]:


model5 = GaussianNB()
model5.fit(X_train,y_train)
prediction5 = model5.predict(X_test)
cm5=confusion_matrix(y_test, prediction5)
cm5


# In[444]:


sns.heatmap(cm5,annot=True, linecolor='White', linewidths=1)
plt.savefig('NaiveBayes.png')


# In[445]:


accuracy_score(y_test, prediction5)


# In[446]:


print(classification_report(y_test, prediction5))


# ## GridSearchCV
# 
# #### Finding the best model and tunning hyper parameters using GridSearchCV

# In[447]:


model_params = {
    'SVM': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20,30,40],
            'kernel': ['rbf','linear']
        }  
    },
    'Random_Forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10,15,20]
        }
    },
    'Logistic_Regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10,15,20]
        }
    },
    'Naive_Bayes_Gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
    'Decision_Tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
            
        }
    }     
}


# In[448]:


from sklearn.model_selection import GridSearchCV
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train,y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df1 = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df1


# **Here, we can see that Random Forest is giving the highest accuracy score with 93% taking "n_estimators" = 15 for predicting breast cancer for a patient.**
