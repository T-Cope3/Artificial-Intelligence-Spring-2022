# Troy Cope
# March 6th, 2022
# Programming Assignment-03-CS4267-Troy Cope

# Importing libraries
# -------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# ---------------------
dataset = pd.read_csv('C:\\Users\\Troy Cope\\PycharmProjects\\Assignment_3_MV\\Multiple-Linear-Dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# -------------------------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
# --------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
# Missing line-----------------------------------------
# Missing line-----------------------------------------
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
# Missing line-----------------------------------------
y_pred = regressor.predict(X_test)


# # Not necessary but printing
# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print(df)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())
