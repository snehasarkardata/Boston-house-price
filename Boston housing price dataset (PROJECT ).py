#!/usr/bin/env python
# coding: utf-8

# In[308]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

#import data modelling libraries

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm


# In[309]:


# Importing the dataset in the variable & assigning the column names
Boston_data= pd.read_csv('housing.csv')
#Boston_data.columns= column_names

# Assigning the name of the column

Boston_data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


# In[310]:


#viewing first 5 rows
Boston_data.head()


# In[311]:


#viewing last 5 rows
Boston_data.tail()


# In[312]:


Boston_data.shape


# In[313]:


Boston_data.isnull().sum()


# In[314]:


# Function to identify numeric features:

def numeric_features(Boston_data):
    numeric_col = Boston_data.select_dtypes(include=np.number).columns.tolist()
    return Boston_data[numeric_col].head()
    
numeric_columns = numeric_features(Boston_data)
print("Numerical Features:")
print(numeric_columns)

print("===="*20)


# In[315]:


# Function to identify categorical features:

def categorical_features(Boston_data):
    categorical_col = Boston_data.select_dtypes(exclude=np.number).columns.tolist()
    return Boston_data[categorical_col].head()

categorical_columns = categorical_features(Boston_data)
print("Categorical Features:")
print(categorical_columns)

print("===="*20)



# Function to check the datatypes of all the columns:

def check_datatypes(Boston_data):
   return Boston_data.dtypes

print("Datatypes of all the columns:")
check_datatypes(Boston_data)


# In[316]:


# Function to detect outliers in every feature

def detect_outliers(df):
    cols = list(df)
    outliers = pd.DataFrame(columns = ['Feature', 'Number of Outliers'])
    for column in cols:
        if column in df.select_dtypes(include=np.number).columns:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            fence_low = q1 - (1.5*iqr)
            fence_high = q3 + (1.5*iqr)
            outliers = outliers.append({'Feature':column, 'Number of Outliers':df.loc[(df[column] < fence_low) | (df[column] > fence_high)].shape[0]},ignore_index=True)
    return outliers

detect_outliers(df)


# In[317]:


# Function to plot histograms

def plot_continuous_columns(Boston_data):
    numeric_columns = Boston_data.select_dtypes(include=['number']).columns.tolist()
    dataframe = Boston_data[numeric_columns]
    
    for i in range(0,len(numeric_columns),2):
        if len(numeric_columns) > i+1:
            plt.figure(figsize=(10,4))
            plt.subplot(121)
            sns.distplot(Boston_data[numeric_columns[i]], kde=False)
            plt.subplot(122)            
            sns.distplot(Boston_data[numeric_columns[i+1]], kde=False)
            plt.tight_layout()
            plt.show()

        else:
            sns.distplot(Boston_data[numeric_columns[i]], kde=False)


# In[318]:


# Function to plot boxplots

def plot_box_plots(Boston_data):
    numeric_columns = Boston_data.select_dtypes(include=['number']).columns.tolist()
    Boston_data = Boston_data[numeric_columns]
    
    for i in range(0,len(numeric_columns),2):
        if len(numeric_columns) > i+1:
            plt.figure(figsize=(10,4))
            plt.subplot(121)
            sns.boxplot(Boston_data[numeric_columns[i]])
            plt.subplot(122)            
            sns.boxplot(Boston_data[numeric_columns[i+1]])
            plt.tight_layout()
            plt.show()

        else:
            sns.boxplot(Boston_data[numeric_columns[i]])

    
    
print("Histograms\n")
plot_continuous_columns(Boston_data)  
print("===="*30)
print('\nBox Plots\n')
plot_box_plots(Boston_data)


# In[319]:


Boston_data.drop(['CHAS'], axis=1, inplace=True)


# In[320]:


Boston_data.shape


# In[321]:


Boston_data.head()


# In[322]:


from scipy.stats.mstats import winsorize

# Function to treat outliers 

def treat_outliers(dataframe):
    cols = list(dataframe)
    for col in cols:
        if col in dataframe.select_dtypes(include=np.number).columns:
            dataframe[col] = winsorize(dataframe[col], limits=[0.05, 0.1],inclusive=(True, True))
    
    return dataframe    


df = treat_outliers(df)

# Checking for outliers after applying winsorization
# We see this using a fuction called 'detect_outliers', defined above.

detect_outliers(df)


# In[323]:


#Prediction of house Price


# Predictors
x = Boston_data.iloc[:,:-1]

# This means that we are using all the columns, except 'MEDV', to predict the house price


# Target
y = Boston_data.iloc[:,-1]

# This is because MEDV is the 'Median value of owner-occupied homes in $1000s'.
# This shows that this is what we need to predict. So we call it the target variable.


# In[324]:


#Feature Selection using Random Forest

def rfc_feature_selection(Boston_data,target):
    X_train, X_test, y_train, y_test = train_test_split(Boston_data, target, test_size=0.3, random_state=42)
    rfc = RandomForestRegressor(random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    rfc_importances = pd.Series(rfc.feature_importances_, index=Boston_data.columns).sort_values().tail(10)
    rfc_importances.plot(kind='bar')
    plt.show()

rfc_feature_selection(x,y)


# In[325]:


x.head(2)


# In[326]:


# Modifying the Predictors to improve the effeciency of the model.

x= x[['CRIM','DIS','RM','LSTAT']]
x.head(2)


# In[327]:


#Scaling the feature variables using MinMaxScaler

mms= MinMaxScaler()
x = pd.DataFrame(mms.fit_transform(x), columns=x.columns)

x.head()


# In[328]:


xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=42)


# In[329]:


#Linear Regression


lr=LinearRegression()

lr.fit(xtrain, ytrain)

coefficients=pd.DataFrame([xtrain.columns, lr.coef_]).T
coefficients=coefficients.rename(columns={0:'Attributes',1:'Coefficients'})
coefficients


# In[330]:


y_pred=lr.predict(xtrain)


# In[331]:


#Model Evaluation


print("R^2: ",metrics.r2_score(ytrain, y_pred))
print("Adusted R^2: ", 1-(1-metrics.r2_score(ytrain, y_pred))*(len(ytrain)-1)/(len(ytrain)-xtrain.shape[1]-1))
print("MAE: ", metrics.mean_absolute_error(ytrain, y_pred))
print("MSE: ", metrics.mean_squared_error(ytrain, y_pred))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(ytrain, y_pred)))

print(metrics.max_error(ytrain, y_pred))


# In[332]:


# visualizing the difference between the actual and predicted price 

plt.scatter(ytrain, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted Vs Actual Prices", fontsize=15)
plt.show()



# In[333]:


#Test data


# Predicting the Test data with model 
ytest_pred=lr.predict(xtest)

lin_acc=metrics.r2_score(ytest, ytest_pred)
print("R^2: ",lin_acc)
print("Adusted R^2: ", 1-(1-metrics.r2_score(ytest, ytest_pred))*(len(ytest)-1)/(len(ytest)-xtest.shape[1]-1))
print("MAE: ", metrics.mean_absolute_error(ytest, ytest_pred))
print("MSE: ", metrics.mean_squared_error(ytest, ytest_pred))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(ytest, ytest_pred)))


# In[334]:


print(metrics.max_error(ytest, ytest_pred))


# In[335]:


# visualizing the difference between the actual and predicted price 

plt.scatter(ytest, ytest_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted Vs Actual Prices", fontsize=15)
plt.show()


# In[336]:


rfr= RandomForestRegressor()

rfr.fit(xtrain, ytrain)

y_pred=rfr.predict(xtrain)


# In[337]:


#Model Evaluation
#Training data


print("R^2: ",metrics.r2_score(ytrain, y_pred))
print("Adusted R^2: ", 1-(1-metrics.r2_score(ytrain, y_pred))*(len(ytrain)-1)/(len(ytrain)-xtrain.shape[1]-1))
print("MAE: ", metrics.mean_absolute_error(ytrain, y_pred))
print("MSE: ", metrics.mean_squared_error(ytrain, y_pred))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(ytrain, y_pred)))

print("\nMaximum Error: ",metrics.max_error(ytrain, y_pred))


# In[338]:


# visualizing the difference between the actual and predicted price 

plt.scatter(ytrain, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted Vs Actual Prices", fontsize=15)
plt.show()


# In[339]:


# Predicting the Test data with model 
ytest_pred=rfr.predict(xtest)

rfr_acc=metrics.r2_score(ytest, ytest_pred)
print("R^2: ",rfr_acc)
print("Adusted R^2: ", 1-(1-metrics.r2_score(ytest, ytest_pred))*(len(ytest)-1)/(len(ytest)-xtest.shape[1]-1))
print("MAE: ", metrics.mean_absolute_error(ytest, ytest_pred))
print("MSE: ", metrics.mean_squared_error(ytest, ytest_pred))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(ytest, ytest_pred)))

print("\nMaximum Error: ",metrics.max_error(ytest, ytest_pred))


# In[340]:


# visualizing the difference between the actual and predicted price 

plt.scatter(ytest, ytest_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted Vs Actual Prices", fontsize=15)
plt.show()


# In[341]:


#3. Support Vector Machine (SVM)

svm_reg=svm.SVR()
svm_reg.fit(xtrain, ytrain)

y_pred=svm_reg.predict(xtrain)


# In[342]:


#Model Evaluation
#Training data

print("R^2: ",metrics.r2_score(ytrain, y_pred))
print("Adusted R^2: ", 1-(1-metrics.r2_score(ytrain, y_pred))*(len(ytrain)-1)/(len(ytrain)-xtrain.shape[1]-1))
print("MAE: ", metrics.mean_absolute_error(ytrain, y_pred))
print("MSE: ", metrics.mean_squared_error(ytrain, y_pred))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(ytrain, y_pred)))

print("\nMaximum Error: ",metrics.max_error(ytrain, y_pred))


# In[343]:


# visualizing the difference between the actual and predicted price 

plt.scatter(ytrain, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted Vs Actual Prices", fontsize=15)
plt.show()


# In[344]:


#Test Data

# Predicting the Test data with model 
ytest_pred=svm_reg.predict(xtest)

svm_acc=metrics.r2_score(ytest, ytest_pred)
print("R^2: ",svm_acc)
print("Adusted R^2: ", 1-(1-metrics.r2_score(ytest, ytest_pred))*(len(ytest)-1)/(len(ytest)-xtest.shape[1]-1))
print("MAE: ", metrics.mean_absolute_error(ytest, ytest_pred))
print("MSE: ", metrics.mean_squared_error(ytest, ytest_pred))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(ytest, ytest_pred)))

print("\nMaximum Error: ",metrics.max_error(ytest, ytest_pred))


# In[345]:


# visualizing the difference between the actual and predicted price 

plt.scatter(ytest, ytest_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted Vs Actual Prices", fontsize=15)
plt.show()


# In[346]:


#Evaluation Comparison of all the 3 methods

models=pd.DataFrame({
    'Model':['Linear Regression', 'Random Forest', 'Support Vector Machine'],
    'R_squared Score':[lin_acc*100, rfr_acc*100,svm_acc*100]
})
models.sort_values(by='R_squared Score', ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




