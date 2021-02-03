#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load all the libraries required for our project
import os
import pandas as pd
import numpy as np
import datetime
import scipy.stats as stats
from pandas import Timestamp
import random as rand
import matplotlib.pyplot as plt
import seaborn as sns
from random import randrange, uniform
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
#libraries for Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error
# libraies of Decision tree model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[2]:


# set working directory
os.chdir("F:/data Scientist/project 1 bike renting/python")
os.getcwd()


# In[3]:


# load data
day= pd.read_csv("day.csv", sep=',')


# In[4]:


day.shape


# In[5]:


# delete the 'instant' variable.
day.head(5)
day=day.drop('instant', axis=1)


# In[6]:


day.dtypes


# In[7]:


#Exploratory Data Analysis
# converting variables to proper data types
day['season']= day['season'].astype(object)
day['yr']= day['yr'].astype(object)
day['mnth']= day['mnth'].astype(object)
day['holiday']= day['holiday'].astype(object)
day['weekday']= day['weekday'].astype(object)
day['workingday']= day['workingday'].astype(object)
day['weathersit']= day['weathersit'].astype(object)

#factor_var=day[['season', 'yr','mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]


# In[8]:


########    VISUALISATIONS for given data  


# In[9]:


# Histograms to see the distribution of data in numerical variables
# Histogram of "temp"
plt.hist(day['temp'], bins=50, color= 'b');
plt.xlabel('temp', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.show()

# Histogram of "atemp"
plt.hist(day['atemp'], bins=50, color= 'b');
plt.xlabel('atemp', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.show()

# Histogram of "hum"
plt.hist(day['hum'], bins=50, color= 'b');
plt.xlabel('hum', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.show()

# Histogram of "windspeed"
plt.hist(day['windspeed'], bins=50, color= 'b');
plt.xlabel('windspeed', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.show()

# Histogram of "Bike rental count"
plt.hist(day['cnt'], bins=50, color= 'b');
plt.xlabel('cnt', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.show()



# from figures it is clear that 'hum' & 'windspeed' data is skewed so it may contain outliers.


# In[10]:


# create bar plots for catagorical variables w.r.t.'cnt'
# plot of 'season' vs 'cnt'
plt.bar(day['season'], day['cnt'], color='g') 
plt.xlabel('season', fontsize=20)
plt.ylabel('count', fontsize=20)
plt.show()

# plot of 'month' vs 'cnt'
plt.bar(day['mnth'], day['cnt'], color='b') 
plt.xlabel('month', fontsize=20)
plt.ylabel('count', fontsize=20)
plt.show()

# plot of 'year' vs 'cnt'
plt.bar(day['yr'], day['cnt'], color='g') 
plt.xlabel('year', fontsize=20)
plt.ylabel('count', fontsize=20)
plt.show()

# plot of 'weekday' vs 'cnt'
plt.bar(day['weekday'], day['cnt'], color='b') 
plt.xlabel('weekday', fontsize=20)
plt.ylabel('count', fontsize=20)
plt.show()

# plot of 'weathersit' vs 'cnt'
plt.bar(day['weathersit'], day['cnt'], color='b') 
plt.xlabel('weathersit', fontsize=20)
plt.ylabel('count', fontsize=20)
plt.show()


# from the bar plots it is clear that high count of rental bikes is seen at season '3' category, and year category '2'


# In[11]:


# scatter plots for numeric variables 
# scatter plot of 'temp' vs 'cnt'
plt.scatter(day['temp'], day['cnt'], marker='o');
plt.xlabel('temp', fontsize=20)
plt.ylabel('cnt', fontsize=20)
plt.show()

# scatter plot of 'atemp' vs 'cnt'
plt.scatter(day['atemp'], day['cnt'], marker='o');
plt.xlabel('atemp', fontsize=20)
plt.ylabel('cnt', fontsize=20)
plt.show()

# scatter plot of 'hum' vs 'cnt'
plt.scatter(day['hum'], day['cnt'], marker='o');
plt.xlabel('hum', fontsize=20)
plt.ylabel('cnt', fontsize=20)
plt.show()

# scatter plot of 'windspeed' vs 'cnt'
plt.scatter(day['windspeed'], day['cnt'], marker='o');
plt.xlabel('windspeed', fontsize=20)
plt.ylabel('cnt', fontsize=20)
plt.show()

# scatter plots of 'hum' and 'windspeed' are not linear.


# In[12]:


####### MISSING VALUE ANALYSIS


# In[13]:


pd.DataFrame(day.isnull().sum())


# In[14]:


#OUTLIER ANALYSIS USING BOXPLOT


# In[15]:


# plotting of box plot for visualising outlier

#   store numeric variables considered for oulier analysis in cnames
cnames= ['temp', 'atemp', 'hum', 'windspeed']

#box plot for numreic varibales


for i in cnames:
    plt.boxplot(day[i])
    plt.title("Boxplot of " +i)
    plt.show()
   


# In[16]:


#create function to calculate atrributes of box plot and replace the outlies with 'na'
def find_outlier(var):

    q75,q25=np.percentile(day[var], [75,25])
   
    iqr= q75-q25
    
    min=q25-(iqr*1.5)
    max=q75+(iqr*1.5)
    
    day.loc[day[var]<min,var]=np.nan
    day.loc[day[var]>max,var]=np.nan
    
   
    
    


# In[17]:


# impute outilers of 'hum' varibale
find_outlier('hum')


# In[18]:


# see the missing valueS These missing values ar enothing but the outliers that we have replaced with .na
day['hum'].isnull().sum()


# In[19]:


# impute na value with mean
day.fillna(day.mean(), inplace=True)
day['hum'].isnull().sum()


# In[20]:


# impute outilers of 'windspeed' varibale
find_outlier('windspeed')
    


# In[21]:


# see the missing valueS 
day['windspeed'].isnull().sum()


# In[22]:


# impute na value with mean
day.fillna(day.mean(), inplace=True)
day['windspeed'].isnull().sum()
day.isnull().sum()
# create copy of data without outliers
day_without_outliers=day


# In[23]:


day_without_outliers.head(5)


# In[24]:


#FEATURE SELECTION


# In[25]:


# Correlation analysis-  it is used to check the dependancies between the variables.
# Correlation plot
cnames_numeric= ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
day_corr= day.loc[:,cnames_numeric]


# In[26]:


# setting the  height and width of plot
f, ax=plt.subplots(figsize=(13,9))

# correaltion matrix
cor_matrix=day_corr.corr()

# plotting correlation plot
sns.heatmap(cor_matrix,mask=np.zeros_like(cor_matrix, dtype=np.bool), 
            cmap=sns.diverging_palette(250,12,as_cmap=True),square=True, ax=ax)


# In[27]:


# form the correlation it is clear that 'temp' and 'atemp' are correlated so will drop 'atemp'


# In[28]:


# ANOVA Test-  it is used to check the dependancies between categorical and numeric variables.
# load statsmodels library to perfom ANOVA test

import statsmodels.api as sm
from statsmodels.formula.api import ols
factor_data=day[['season', 'mnth','yr','holiday', 'weekday', 'workingday', 'weathersit']]

#perform the anova test 
for i in factor_data:
    
    print(i)
    
    factor_anova=ols('cnt~factor_data[i]', data=day).fit()
    aov_table= sm.stats.anova_lm(factor_anova, type=2)
    
    print(aov_table)


# In[29]:


# if we see the output of anova test it is clear that 'holiday', 'weekday' and 'workingday' are having the P value<0.05
# so we can remove these variables. we will perfom chi square test on these variables again


# In[30]:


# Chi square test-  it is done on cateorical variables onl to check the dependancies between them.
# chi square test of independance
from scipy.stats import chi2_contingency

factor_data=day[['season', 'mnth','yr','holiday','weekday','workingday', 'weathersit']]
for i in factor_data:
    for j in factor_data:
        if(i!=j):
            chi2,p,dof, ex=chi2_contingency(pd.crosstab(day[i],day[j]))
            while(p<0.05):
                print(i)
                print(j)
                print(p)
                break
        
        


# In[31]:


# from the above output of chi square test it shows the variables with p value <0.05 
# so common variables are 'holiday', 'weekday' and 'workingday' so we will skip these variables.


# In[32]:


day=day_without_outliers
day.head(5)


# In[33]:


# removing the correlated varibales ( selecting only relevant variables w.r.t. target variable)

day_deleted= day.drop(['atemp', 'casual', 'registered', 'holiday','weekday', 'workingday','dteday'],axis=1)
day_deleted.head(5)


# In[34]:


# PREPARING THE MACHINE LEARNING MODELS FOR OUR DATA
# As our target variable is numeric so we will select regression models 
# I have selected the following three models for this project
# I will check the accuracy of three models and select the best one
# 1)Decision tree 
# 2)Linear regression 
# 3)Random forest


# In[35]:


# DECISION TREE REGRESSION MODEL
# this model creates the decision tree like flow chart and creates the rules to predict the target variable.


# In[36]:


# divide the data into train and test
train, test=train_test_split(day_deleted, test_size=0.2)


# In[37]:


# create decision tree for regression
fit=DecisionTreeRegressor(max_depth= 2).fit(train.iloc[:,0:7], train.iloc[:,7])


# In[38]:


# apply model on test data
predictions_dt=fit.predict(test.iloc[:,0:7])


# In[39]:


# define the funtion to find MAPE for our model
import numpy as np
def MAPE(y_true, y_pred):
    mape= np.mean(np.abs((y_true-y_pred)/y_true))*100
    return mape


# In[40]:


# MAPE- mean absolute percentage error- it provides the error between real and predicted values 
# so accuracy of model is 100- MAPE
# calculate MAPE

MAPE(test.iloc[:,7], predictions_dt)


# In[41]:


#calculate MAE MSE and RMSE
print('Mean Absolute Error:',metrics.mean_absolute_error(test['cnt'], predictions_dt))
print('Mean Squared Error:', metrics.mean_squared_error(test['cnt'], predictions_dt))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(test['cnt'], predictions_dt)))


# In[42]:


#Predictive performance of model using error metrics 

#MAPE:26.96%
#Accuracy: 73.04%
#MAE: 1958.58
#MSE:5621933.20
#RMSE:2371.06


# In[43]:


# if we see the performace metrics of decision tree results are quite good but needs improvements.


# In[44]:


#create dataframe of real and predicted values
Result_dt=pd.DataFrame({'real':test.iloc[:,7],'predicted': predictions_dt})
Result_dt


# In[45]:


# LINNEAR REGRESSION MODEL
# this model is only used for numeric target variable
# it creates the coefficients of predictor variable w.r.t. target variable.


# In[46]:


# Training the model 

train, test=train_test_split(day_deleted, test_size=0.2)


# In[47]:


# build linear Regression model
lr_model=sm.OLS(train.iloc[:,7], train.iloc[:,0:7].astype(float)).fit()


# In[48]:


lr_model.summary()


# In[49]:


# if we see the liner regressionmodel summary above we can see that adj r squared value is 96.5% whcih is good
# also F-statistic is far greater than 1 and P value<0.05
# model is good but still we need to improve the accuracy 


# In[50]:


test.shape


# In[51]:


# do the predictions
predictions_lr=lr_model.predict(test.iloc[:,0:7])


# In[52]:


MAPE(test.iloc[:,7], predictions_lr)


# In[53]:


# calculate MAE MSE and RMSE 
print('Mean Absolute Error:',metrics.mean_absolute_error(test['cnt'], predictions_lr))
print('Mean Squared Error:', metrics.mean_squared_error(test['cnt'], predictions_lr))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(test['cnt'], predictions_lr)))


# In[54]:


#Predictive performance of model using error metrics 

#MAPE:22.76%
#Accuracy: 77.24%
#MAE:709.16
#MSE:912880.30
#RMSE:955.44


# In[55]:


# the accuracy of this model is good as compared to decision tree model.


# In[56]:


#create dataframe of real and predicted values
Result_lr=pd.DataFrame({'real':test.iloc[:,7],'predicted': predictions_lr})
Result_lr


# In[57]:


########### RANDOM FOREST MODEL
# this is the improved version of decision tree model it uses many trees in one model to improve the accuracy.
# it feeds error of one tree to another tree to improve the accuracy.


# In[58]:


# divide the data into train and test
x=day_deleted.values[:,0:7]
y=day_deleted.values[:,7]

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2,random_state=0)


# In[59]:


# building a random forest model
rf_model= RandomForestRegressor(n_estimators=70, random_state=0)
rf_model.fit(x_train, y_train)


# In[60]:


# applying the model to predict cnt on test data
predictions_rf= rf_model.predict(x_test)
predictions_rf


# In[61]:


# calculate MAE MSE and RMSE for our model
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, predictions_rf))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions_rf))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, predictions_rf)))


# In[62]:


def MAPE(y_true, y_pred):
    mape= np.mean(np.abs((y_true-y_pred)/y_true))*100
    return mape


# In[63]:


MAPE(y_test, predictions_rf)


# In[64]:


#Predictive performance of model using error metrics 

#MAPE:20.21%
#Accuracy: 79.79%
#MAE: 535.24
#MSE: 479937.73
#RMSE:692.77


# In[65]:


#########  chosing the model among 3 models we have used 
#From the Error metrics of all three models we can say that Random Forest model is having high accuracy
# Other error metrics like MAE MSE and RMASE values are also less than other 2 models 
# so we will choose Random Forest as our output model of this project to predict the bike rental count. 


# In[66]:


# selecting the RF model as output model of this project
rf_model_sel= rf_model


# In[67]:


#Predicting the 'cnt' for whole data using RF model
predictions_rf_out= rf_model_sel.predict(x)
predictions_rf_out


# In[68]:


# creating the dataframe of real and predicted values of cnt for whole data
Result__rf_out=pd.DataFrame({'real':y,'predicted': predictions_rf_out})
Result__rf_out


# In[69]:


# Adding the predicted values in data 
day_deleted.insert(8,"predicted", predictions_rf_out, True)


# In[70]:


day_deleted.head(5)


# In[71]:


day.shape


# In[72]:


# line plot of real and predicted values of cnt
plt.figure(figsize=(15,8))
plt.plot(y,color= 'g')
plt.xlabel('real= Green & predicted= Red', fontsize= '20')
plt.title= ("whadfhkafafka")
plt.plot(predictions_rf_out, color='r')
plt.show()
#plt.xlabel('Graph of real and predicted values of cnt')


# In[73]:


# the graph showing real and predicted values by RF model is looking pretty good their is less variation in both the values.


# In[74]:


# scatter plot or Real and Predicted values of Bike rent count.
plt.scatter(y,predictions_rf_out,alpha=0.5)
plt.xlabel('Real', fontsize=20)
plt.ylabel('Predicted', fontsize=20)
plt.show()


# In[75]:


# scatter plot of real and predicted values is also showing linear relationship.
# so we can say that Random Forest model is best for our study.


# In[76]:


# Adding the predicted values in  main data 
day.insert(15,"predicted", predictions_rf_out, True)


# In[77]:


# Now we will save the output of model as a 'python_output_main' in the data we selected after preprocessing


# In[78]:


day_deleted.to_csv("python_out_preprocessed.csv", index=False)


# In[79]:


# Now we will save the output as 'python_output.csv' fpr the main data  
day.to_csv('python_output_main.csv', index= True)

