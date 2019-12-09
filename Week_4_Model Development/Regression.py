import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt

path = "automobileEDA.csv"
df = pd.read_csv(path)

lr = LinearRegression()

# How could Highway-mpg help predict car price
X = df[["highway-mpg"]]
Y = df[["price"]]
lr.fit(X, Y)

print("coefficient is: ", lr.coef_)
print("intercept is ", lr.intercept_)

Yhat = lr.predict(X)

print("predicted value of price ", Yhat[0:5])

# Develop a model using horsepower, curb-weight, engine-size, highway-mpg variables as the predictor variables
X1 = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
Y1 = df[["price"]]

lr1 = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
lr1.fit(X1, Y1)

print("intercept ", lr1.intercept_, " coefficient ", lr1.coef_)

# visualize horsepower as potential predictor variable of price
# using regression plot

#sns.regplot(df["highway-mpg"], df["price"])
#plt.ylim(0,)
#plt.title("Regression plot of highway-mpg and price")
#plt.savefig("RegPlot_highway_mpg")
#plt.show()

# regression plot of "peak-rpm" LR not the right model since large variance
#sns.regplot(x="peak-rpm", y="price", data=df)
#plt.ylim(0,)
#plt.savefig("RegPlot_peak_rpm")


# Is "peak-rpm" or "highway-mpg" more strongly correlated with "price" using corr()
df_corr = df[["peak-rpm", "highway-mpg", "price"]]
print("Correlation matrix \n", df_corr.corr())
# highway-mpg = -0.704692; peak-rpm = -0.101616

# residual plot highway-mpg and price; residuals are not randomly spread around the x-axis
#sns.residplot(x="highway-mpg", y="price", data=df)
#plt.savefig("residual_plot")
#plt.show()


# Distribution plot for multiple linear regression
lr = LinearRegression()
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y0 = df["price"]
lr.fit(Z, Y0)
Y1 = lr.predict(Z)

ax1 = sns.distplot(Y0, hist=False, label="Actual Value", color="b")
sns.distplot(Y1, hist=False, label="Predicted Value", color="r", ax=ax1)
plt.xlabel("Price")
plt.ylabel("Percentage of cars")
plt.title("Distribution Plot")
plt.xlim(-10000,)
plt.savefig("Distribution Plot")
plt.show()
plt.close()

# using highway-mpg as the predictor variable, try fitting a polynomial model to the data instead

poly = PolynomialFeatures(degree=2)
Zbar = poly.fit_transform(Z)
print("Before and after transform ", Z.shape," ", Zbar.shape)

# create a pipeline (normalize, transform and use linear regression)
#scale = StandardScaler()
#pf = PolynomialFeatures(degree=2)
#lrmodel = LinearRegression()
pipe_input = [("scale", StandardScaler()), ("pf", PolynomialFeatures(degree=2)), ("lrmodel", LinearRegression())]

pipe = Pipeline(pipe_input)
#print(pipe)
pipe.fit(Z, df["price"])
Yhat_pipeline = pipe.predict(Z)
print("Prediction using pipeline \n", Yhat_pipeline[0:4])

# simple linear regression
# R-score of highway-mpg and price fit
lr.fit(X, Y)
print("R-squared value of SLR: ", lr.score(X, Y))

# MSE calculation
print("Mean-squared error of SLR: ", metrics.mean_squared_error(df["price"], lr.predict(X)), "\n")

# Multiple linear regression
# R-score of variables 'horsepower', 'curb-weight', 'engine-size', 'highway-mpg' and 'price' fit
Y = df["price"]
lr.fit(Z, Y)
print("R-squared value of MLR: ", lr.score(Z, Y))
# ~80% of the variation of price is explained by the multiple linear regression fit

# MSE of variables 'horsepower', 'curb-weight', 'engine-size', 'highway-mpg' and 'price' fit
Yhat = lr.predict(Z)
print("MSE value of MLR: ", metrics.mean_squared_error(Y, Yhat), "\n")

# Polynomial regression

# R2 and MSE
pf = PolynomialFeatures(degree=2)
x = df["highway-mpg"]
y = df["price"]

# polyfit fits a polynomial of degree deg to points (x, y).
# Returns a vector of coefficients p that minimises the squared error
f = np.polyfit(x, y, 3)

# to construct the polynomial
p = np.poly1d(f)

print("R-score value for Polynomial regression: ", metrics.r2_score(y, p(x)))
print("MSE value for Polynomial regression: ",metrics.mean_squared_error(df["price"], p(x)))


# create an array of evenly spaced values
np_array = np.arange(1, 100, 1)
print("np-array: ", np_array, " ", np_array.shape)
values = np_array.reshape(-1, 1)
print("np-array after reshaping: ", values.shape)
X = df[["engine-size"]]
Y = df[["price"]]
lr.fit(X, Y)

Yhat = lr.predict(values)
plt.plot(values, Yhat)
plt.savefig("sample_plot")
plt.show()

