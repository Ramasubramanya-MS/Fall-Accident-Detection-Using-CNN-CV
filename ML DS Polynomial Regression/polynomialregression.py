#!/usr/bin/env python
# coding: utf-8

# In[236]:


import csv
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[237]:


##Choose .csv file from computer
from google.colab import files
uploaded = files.upload()


# In[276]:


import csv
filename = "DS.csv"

xval = []
yval = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)

    for row in csvreader:
        xval.append(float(row[0]))
        yval.append(float(row[1]))
        
df = pd.read_csv('DS.csv')
plt.scatter(xval, yval)
plt.show()


# In[277]:


import numpy
mymodel = numpy.poly1d(numpy.polyfit(xval, yval, 3))

myline = numpy.linspace(1, 22, 100)

plt.scatter(xval, yval)
plt.plot(myline, mymodel(myline))
plt.show()


# In[277]:





# In[278]:


xval[:10]


# In[279]:


yval[:10]


# In[280]:


df.corr(method ='pearson')


# In[281]:


df.corr(method ='kendall')


# In[282]:


plt.scatter(df['-12.0'], df['0.0'])


# In[283]:


data = pd.read_csv('DS.csv')
x = df['-12.0'].values
y = df['0.0'].values


# In[284]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[285]:


x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)


# In[286]:


y_train = y_train[x_train[:,0].argsort()]
x_train = x_train[x_train[:, 0].argsort()]


# In[287]:


X = np.array(df['-12.0']).reshape(-1, 1)
y = df['0.0']
lin_reg = LinearRegression()
lin_reg.fit(X, y)

plt.scatter(X, y)
plt.plot(x, lin_reg.predict(X), color='red')
##The straight line does not acuurately fit the curve so we have to choose higher order equations


# In[288]:


#plt.title('Linear Regression')
#plt.xlabel('-12.0')
#plt.ylabel('0.0')
#plt.scatter(x, y)
#plt.plot(x_train, poly_reg.predict(poly1), c='red', label='Polynomial regression line')
#plt.legend(loc="upper left")
#plt.show()


# In[289]:


poly2 = PolynomialFeatures(degree=2)
x_poly2 = poly2.fit_transform(x_train)
poly_reg = LinearRegression()
poly_reg.fit(x_poly2, y_train)


# In[290]:


plt.title('2nd order regression')
plt.xlabel('-12.0')
plt.ylabel('0.0')
plt.scatter(x, y)
plt.plot(x_train, poly_reg.predict(x_poly2), c='red', label='Polynomial regression line')
plt.legend(loc="upper left")
plt.show()
##the curve is not fitting properly so we have to choose a higher order function


# In[291]:


poly3 = PolynomialFeatures(degree=3)
x_poly3 = poly3.fit_transform(x_train)
poly_reg = LinearRegression()
poly_reg.fit(x_poly3, y_train)


# In[292]:


plt.title('3rd order regression')
plt.xlabel('-12.0')
plt.ylabel('0.0')
plt.scatter(x, y)
plt.plot(x_train, poly_reg.predict(x_poly3), c='red', label='Polynomial regression line')
plt.legend(loc="upper left")
plt.show()
##the curve fits exactly so the 3rd order regression works properly


# In[293]:


lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
print('Coefficients of x are', lin_reg.coef_)
print('Intercept is', lin_reg.intercept_)


# In[299]:


print("Decision Tree Regressor:")


# In[300]:


import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeRegressor
import matplotlib.image as pltimg

df = pandas.read_csv(filename, header=None)

print(df)


# In[301]:


x = [[i] for i in df[0]]
y = [i for i in df[1]]

print(x)
print(y)


# In[302]:


rng = numpy.random.RandomState(1)
X = numpy.sort(5 * rng.rand(80, 1), axis=0)
y = numpy.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))


# In[303]:


X = df[0].values
y = df[1].values


# In[304]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


# In[305]:


treeRegr = DecisionTreeRegressor(max_depth=10)
treeRegr.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))


# In[306]:


y_pred = treeRegr.predict(X_test.reshape(-1,1))


# In[307]:


r2 = r2_score(y_test, y_pred)
print("r square value for Decision Tree Regression:",r2)


# In[308]:


X_grid = numpy.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'r')
plt.scatter(X_test, y_pred, color = 'g')
plt.title('Decision Tree Regression')
plt.show()

plt.plot(X_grid, treeRegr.predict(X_grid), color = 'k')
plt.title('Decision Tree Regression')
plt.show()


# In[309]:


treeRegr.fit(X_train.reshape(-1,1), y_train.reshape(-1,1)).score(X_train.reshape(-1,1), y_train.reshape(-1,1))


# In[311]:


print("Random Forest Regression")


# In[312]:


import pandas
import numpy
from sklearn import tree
import pydotplus
import matplotlib.image as pltimg
filename="DS.csv"
df = pandas.read_csv(filename, header=None)

print(df)


# In[313]:


x = [[i] for i in df[0]]
y = [i for i in df[1]]


# In[314]:


print(x)
print(y)


# In[315]:


rng = numpy.random.RandomState(1)
X = numpy.sort(5 * rng.rand(80, 1), axis=0)
y = numpy.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))
X = df[0].values
y = df[1].values

print(X)
print(y)


# In[316]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


# In[317]:


# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor
  
 # create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
  
# fit the regressor with x and y data
regressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))  


# In[318]:


from sklearn.metrics import r2_score
y_pred = regressor.predict(X_test.reshape(-1,1))
r2 = r2_score(y_test, y_pred)
print("r square value for Random Forest Regression:",r2)


# In[319]:


X_grid = numpy.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color = 'r')
plt.scatter(X_test, y_pred, color = 'g')
plt.title('Random Forest Regression')
plt.show()

plt.plot(X_grid, regressor.predict(X_grid), color = 'k')
plt.title('Random Forest Regression')
plt.show()


# In[320]:


print("SVR regression")


# In[321]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X.reshape(-1,1))
y = sc_y.fit_transform(y.reshape(-1,1))


# In[322]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[323]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))


# In[324]:


y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)


# In[325]:


y_pred


# In[326]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X.reshape(-1,1))
y = sc_y.fit_transform(y.reshape(-1,1))


# In[327]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[328]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))


# In[329]:


y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)


# In[330]:


r2 = r2_score(y_test, y_pred)
print("r square value for SVR:",r2)


# In[186]:


X_grid = numpy.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X_test), sc_y.inverse_transform(y_test.reshape(-1)), color = 'red')
plt.scatter(sc_X.inverse_transform(X_test), y_pred, color = 'green')
plt.title('SVR Regression')
plt.show()


# In[1]:


pip install nbconvert


# In[187]:




