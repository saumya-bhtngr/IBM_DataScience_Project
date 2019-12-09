import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from matplotlib import pyplot as plt


path = "module_5_auto.csv"
df = pd.read_csv(path)

# store dependent variable in Y and independent variables in X
Y = df[["price"]]
X = df.drop("price", axis=1)

# split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

print("Training data size: ",X_train.shape[0])     # (170, 30)
print("Test data size: ",X_test.shape[0])          # (31, 30)


# fit the model using feature "horsepower and find R^2
lr = LinearRegression()
lr.fit(X_train[["horsepower"]], Y_train)

# calulate the R^2 value
print("R^2 value on test data", lr.score(X_test[["horsepower"]], Y_test))

print("R^2 value on train data", lr.score(X_train[["horsepower"]], Y_train))


# Cross validation score

R_cross = cross_val_score(lr, X[["horsepower"]], Y, cv=4)
print("Cross validation R^2 values: ", R_cross)

# get the mean and std of R_cross
print("Mean and STD of R_cross: ", R_cross.mean(), " ", R_cross.std())

# Cross validation predict
predict_cv = cross_val_predict(lr, X[["horsepower"]], Y, cv=4)
print("Prediction values after cross validation \n", predict_cv[0:4])


# MLR model using 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg' as features
mlr = LinearRegression()
x_train_data = X_train [["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
#y_data = df[["price"]]

mlr.fit(x_train_data, Y_train)

# prediction using train and test data
Yhat_train = mlr.predict(x_train_data)
print("prediction values from train data ", Yhat_train[0:5])

Yhat_test = mlr.predict(X_test[["horsepower", "curb-weight", "engine-size", "highway-mpg"]])
print("prediction values from test data ", Yhat_test[0:5])

# Examine the distribution of the predicted values of the training data
ax1 = sns.distplot(Y_train, hist=False, color="b", label="Actual values train data")
sns.distplot(Yhat_train, hist=False, color="r", label="Predicted values test data", ax=ax1)
plt.savefig("MLR_distribution_train")
plt.close()
#plt.show()

# Examine the distribution of the predicted values of the test data
ax2 = sns.distplot(Y_test, hist=False, color="b", label="Actual values test data")
sns.distplot(Yhat_test, hist=False, color="r", label="Predicted values test data", ax=ax2)
plt.savefig("MLR_distribution_test")
plt.show()
plt.close()

# Create a degree 5 polynomial model. Use 55 percent of the data for testing and the rest for training
# 5 polynomial transformation on the feature 'horse power'.
poly = PolynomialFeatures(degree=5)
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(X, Y, test_size=0.45, random_state=0)

x_train_fit = poly.fit_transform(x_train_2[["horsepower"]])
x_test_fit = poly.fit_transform((x_test_2[["horsepower"]]))


# Create a linear regression model "poly" and train it
poly = LinearRegression()
poly.fit(x_train_fit, y_train_2)

# Get predicted values of price column
Yhat = poly.predict(x_test_fit)
print("Predicted values from polynomial regression :\n", Yhat[0:5])

# Take the first five predicted values and compare it to the actual targets
print("Actual values :\n", y_test_2[0:5])

# Get R^2 of training and test data
R_sq_train_poly = poly.score(x_train_fit, y_train_2)
R_sq_test_poly = poly.score(x_test_fit, y_test_2)
print("Polynomial Regression R^2 values ",R_sq_train_poly," and ",R_sq_test_poly)


# How the R^2 changes on the test data for different order polynomials and plot the results
orders = [1, 2, 3, 4, 5, 6]
R_sq = []
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
for order in orders:
    polyF = PolynomialFeatures(degree=order)
    X_train_fit = polyF.fit_transform(X_train[["horsepower"]])
    X_test_fit = polyF.fit_transform(X_test[["horsepower"]])
    lr.fit(X_train_fit, Y_train)
    R_sq_value = lr.score(X_test_fit, Y_test)
    R_sq.append(R_sq_value)

print("R-sq values by varying degree of polynomial are: ", R_sq)

# plot order versus values in R_sq list
plt.plot(orders, R_sq)
plt.xlabel("Degree")
plt.ylabel("R-squared Value")
plt.title("R-sq_changes_vs_degree")
plt.savefig("R-sq_changes_vs_degree")
plt.close()


# We can perform polynomial transformations with more than one feature
# Create a "PolynomialFeatures" object "pr1" of degree two?

pr1 = PolynomialFeatures(degree=2)

# Transform the training and testing samples for the features 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg'
train_transformed_features = pr1.fit_transform(X_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
test_transformed_features = pr1.fit_transform(X_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print("Shape of transformed feature:",train_transformed_features.shape, test_transformed_features.shape)

# Create a linear regression model "poly1" and train the object using the method "fit" using the polynomial features
poly1 = LinearRegression()
poly1.fit(train_transformed_features, Y_train)

# Use the method "predict" to predict an output on the polynomial features,
# then use the function "DistributionPlot" to display the distribution of the predicted output vs the test data

Yhat_poly = poly1.predict(test_transformed_features)
ax_poly = sns.distplot(Y_test, hist=False, color="b", label="Test Data Actual Price")
sns.distplot(Yhat_poly, hist=False, color="r", label="Test Data Predicted Price", ax=ax_poly)
plt.savefig("distribution_polynomial_features")
plt.show()
plt.close()






















