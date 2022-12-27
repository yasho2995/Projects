#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[6]:


boston = load_boston()
boston.data.shape


# In[8]:


feature_num = 7
boston = load_boston()
X = boston.data[:,:feature_num]
y = boston.target
features = boston.feature_names[:feature_num]
pd.DataFrame(X, columns= features).head()


# In[9]:


y[0:5]


# In[11]:


# split into training and testing sets and standardize them
X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=1)
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)


# In[15]:


# loop through different penalty score (alpha) and obtain the estimated coefficient (weights)
alphas = 10 ** np.arange(1, 5)
print('different alpha values:', alphas)

# stores the weights of each feature
ridge_weight = []
for alpha in alphas:    
    ridge = Ridge(alpha = alpha, fit_intercept = True)
    ridge.fit(X_train_std, y_train)
    ridge_weight.append(ridge.coef_)


# In[16]:


alphas


# In[17]:


ridge_weight


# In[21]:


def weight_versus_alpha_plot(weight, alphas, fratures):
    """Pass in the estimated weight, the alpha value and the names
    for the features and plot the model's estimated coefficient 
    weight for different alpha values
    """
    
    fig = plt.figure(figsize=(8,6))
    
    # the weight should be array
    weight = np.array(weight)
    for col in range(weight.shape[1]):
        plt.plot(alphas, weight[:,col], label = features[col])
        
    plt.axhline(0, color = 'black', linestyle = '--', linewidth = 3)
    
    # specify the coordinates
    plt.legend(bbox_to_anchor = (1.3,0.9))
    plt.title('Coefficient weight as Alpha Grows')
    plt.ylabel('Coefficient weight')
    plt.xlabel('alpha')
    return fig


# In[22]:


# change default figure and font size
plt.rcParams['figure.figsize'] = 8, 6 
plt.rcParams['font.size'] = 12


ridge_fig = weight_versus_alpha_plot(ridge_weight, alphas, features)


# In[25]:


# does the same thing above except for lasso
alphas = [0.01, 0.1, 1, 5, 8]
print('different alpha values:', alphas)

lasso_weight = []
for alpha in alphas:    
    lasso = Lasso(alpha = alpha, fit_intercept = True)
    lasso.fit(X_train_std, y_train)
    lasso_weight.append(lasso.coef_)

lasso_fig = weight_versus_alpha_plot(lasso_weight, alphas, features)    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




