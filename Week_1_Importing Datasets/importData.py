import pandas as pd

path = "imports-85.data"

# read and save data into a data frame
df = pd.read_csv(path, header=None)

# print top 10 and bottom 10 rows
print('top 10 rows \n', df.head(10))

print('bottom 10 rows \n', df.tail(5))

# add headers
header = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df.columns = header

# print top 10 rows with added headers
print('rows with added headers \n', df)

# drop missing values from price column
df.dropna(subset=["price"], axis=0)
print('rows after removing missing values \n', df)

# find the name of the columns
print ("column names are: ", df.columns)

# save the data frame; index=False for no row names
df.to_csv("exported_automobile_data.csv", index=False)

# check data types
print(df.dtypes)

# get statistical summary of each column; include="all" for object types summary
print(df.describe(include="all"))

# select only some columns and get their summary
print(df[["symboling", "highway-mpg"]].describe())

print("summary using info method \n", df.info)



