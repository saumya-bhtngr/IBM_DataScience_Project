import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "/Users/SaumyaBhatnagar/PycharmProjects/IBM_DataScience_Project/auto.csv"
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


df = pd.read_csv(path)
df.columns = headers

# replace ? with nan
df.replace("?", np.nan, inplace=True)

# find out how many ? values for column horsepower in the dataset
missing_data = df.isnull()
print(missing_data)
for column in df.columns.values.tolist():
    if column == "horsepower":
        print("before replacing nan", missing_data[column].value_counts())


mean_horsepower = df["horsepower"].astype("float").mean()
# replace nan with mean
print("mean horsepower ", mean_horsepower)
df["horsepower"].replace(np.nan, mean_horsepower, inplace=True)

for column in df.columns.values.tolist():
    if column == "horsepower":
        print("after replacing nan", df[column].value_counts())

df["horsepower"] = df["horsepower"].astype(int, copy=True)

# show histogram without binning
#plt.hist(df["horsepower"])

#plt.xlabel("Horsepower")
#plt.ylabel("Frequency")
#plt.title("Histogram of car horsepower distribution")
#plt.savefig("horsepower_histogram.png")
#plt.show()

print("completed execution ")

# three categories or bins for horsepower; so 4 dividers
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)

group_names = ["low", "medium", "high"]

df["horsepower_binned"] = pd.cut(df["horsepower"], bins, labels=group_names, include_lowest=True)

# display first 20 values with categorization
print(df[["horsepower", "horsepower_binned"]].head(20))
# get the value count of each bin
print(df["horsepower_binned"].value_counts())

# plot a bar graph
plt.bar(group_names, df["horsepower_binned"].value_counts())
plt.xlabel("Horsepower")
plt.ylabel("Value Counts")
plt.title("Bar graph of car horsepower distribution")
plt.savefig("barGraph_horsepower.png")
plt.show()

# show the histogram with binning
plt.hist(df["horsepower_binned"], bins=3)
plt.xlabel("Horsepower")
plt.ylabel("Frequency")
plt.title("Histogram of car horsepower distribution")
plt.savefig("histogram_horsepower.png")
plt.show()

# indicator or dummy variables (for example: to use categorical data in regression)
# convert "fuel-type" into indicator variables.
# get indicator variables and assign it to data frame "dummy_variable_1"
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
print(dummy_variable_1.head())

# dummy_variable_1.rename(columns={"fuel_type_diesel":"diesel", "fuel_type_diesel":"gas"}, inplace=True)
dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
print(dummy_variable_1.head())

print("original data frame \n", df.head())                                    # [5 rows x 27 columns]
# insert new columns into data frame
df = pd.concat([df, dummy_variable_1], axis=1)
print("after inserting two new columns \n", df.head())                        # [5 rows x 29 columns]

# drop original column fuel-type from data frame
df.drop("fuel-type", axis=1, inplace=True)                                    # [5 rows x 28 columns]

print("after dropping a column \n", df.head())

df.to_csv("clean_data.csv")