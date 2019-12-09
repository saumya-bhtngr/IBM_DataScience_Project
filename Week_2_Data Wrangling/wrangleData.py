import pandas as pd
import numpy as np

path = "/Users/SaumyaBhatnagar/PycharmProjects/IBM_DataScience_Project/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df = pd.read_csv(path, names=headers)
print("first 5 rows \n", df.head(20))

# replace ? to NaN
df.replace("?", np.nan, inplace=True)
print("first 20 rows after replacemnet \n", df.head(20))

# check for missing data
missing_data = df.isnull()
print("missing data \n", missing_data)
#print(missing_data.shape)

# count missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

# calculate average of column normalized-losses; without converting to float will give error
average = df["normalized-losses"].astype("float").mean()
print("average of column normalized losses is: ", average)

# replace nan values in "normalized-losses" column with average
df["normalized-losses"].replace(np.nan, average, inplace=True)
print(df)

# see which values are present in a particular column "no of doors"
print("Num of doors: \n", df["num-of-doors"].value_counts())

print("Most frequently ocurring number of doors values: ", df["num-of-doors"].value_counts().idxmax())

# replace the missing 'num-of-doors' values by the most frequent value
mode_value = df["num-of-doors"].value_counts().idxmax()
print("Mode of number of doors variable: ", mode_value)
print("before replacing nan values ", df["num-of-doors"])
df["num-of-doors"].replace(np.nan, mode_value, inplace=True)
print("after replacing nan values ", df["num-of-doors"])



# drop all rows that do not have price data
print("before dropping out rows: ", df.shape)          # (205, 26)
df.dropna(subset=["price"], axis=0, inplace=True)
print("after dropping out rows: ", df.shape)           # (201, 26)


# reset index, because we dropped two rows
print("before re-indexing \n", df)
df.reset_index(drop=True, inplace=True)
print("after re-indexing \n", df)

# list the data types for each column
print("\n original data types \n", df.dtypes)

# change data type of "bore" and "stroke" to float
df["bore"] = df["bore"].astype("float")
df["stroke"] = df["stroke"].astype("float")
df["price"] = df["price"].astype("float")
df["normalized-losses"] = df["normalized-losses"].astype("int")
df["peak-rpm"] = df["peak-rpm"].astype("float")

print("\n after changing data types \n", df.dtypes)

# data transformation to transform mpg into L/100km
print("before transformation \n", df["city-mpg"])
df["city-L/100km"] = 235/df["city-mpg"]
print("after transformation \n", df["city-L/100km"])

# normalize the column "height"
print("before normalizing \n", df["height"])
df["height"] = df["height"]/df["height"].max()
print("after normalizing \n", df["height"])













