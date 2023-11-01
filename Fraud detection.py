#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

from plot_confusion_matrix import plot_confusion_matrix


# In[47]:


data=pd.read_csv('data.csv')


# In[48]:


data.head()


# In[49]:


data.describe().round(decimals=2)


# In[50]:


n_genuine = len(data[data['Class'] == 0])
n_fraud = len(data[data['Class'] == 1])                


# In[51]:


print('Number of Genuine Transactions : ', n_genuine)
print('Number of Fraud Transactions : ', n_fraud)

plt.pie([n_genuine, n_fraud], labels=['Genuine', 'Fraud'], radius=1)
plt.show()


# In[52]:


X, y = data.iloc[:, :-1], data.iloc[:, -1]
X.head()


# In[53]:


k = 10
k_best = SelectKBest(f_classif, k=k)
k_best.fit(X, y)


# In[54]:


mask = k_best.get_support()
not_mask = np.logical_not(mask)

all_features = np.array(list(X))

best_features = all_features[mask]
bad_features = all_features[not_mask]

print('Best Features : ', best_features)
print('Bad Features : ', bad_features)


# In[55]:


X = X.drop(bad_features, axis=1)
X.head()


# In[56]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[57]:


nb = GaussianNB() 
cv_results = cross_validate(nb, x_train, y_train, cv=10, 
                            scoring='recall', 
                            return_train_score=True,
                            return_estimator=True)


# In[58]:


print('training score from each fold: ',cv_results['train_score'])
max_score_index = np.argmax(cv_results['train_score'])
best_estimator = cv_results['estimator'][max_score_index]


# In[59]:


def display_results(estimator, x, y):
    predicted = estimator.predict(x)
    cm = confusion_matrix(y, predicted)
    report = classification_report(y, predicted)
    print(report)
    plot_confusion_matrix(cm, classes=['Genuine', 'Fraud',], title='Fraud Detection')
    


# In[60]:


display_results(best_estimator, x_test, y_test)


# In[61]:


display_results(best_estimator, x_train, y_train)


# In[ ]:





# In[ ]:




