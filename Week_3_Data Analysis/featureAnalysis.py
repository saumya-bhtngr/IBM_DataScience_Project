import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy as scp

path = "automobileEDA.csv"

df = pd.read_csv(path)

print(df.head)                   # [201 rows x 29 columns]

# list the data types for each column
print(df.dtypes)

# list the data types for column peak-rpm
print("Data type of column peak-rpm is: ", df["peak-rpm"].dtypes)

# find the correlation between the following columns: bore, stroke, compression-ratio, and horsepower
print(df[["bore", "stroke", "compression-ratio", 'horsepower']].corr())

# find the scatterplot (with regression line) of "engine-size" and "price"
sns.regplot(x = "engine-size", y = "price", data = df)
plt.title("Scatterplot of engine-size and price")
plt.savefig("scatter plot of size and price")
#plt.show()


print("correlation between engine-size and price is \n", df[["engine-size", "price"]].corr())   # 0.872

# examine the correlation between 'highway-mpg' and 'price'
sns.regplot(x='highway-mpg', y='price', data=df)
plt.savefig("scatter plot of highway-mpg and price")
plt.title("Scatter plot of highway-mpg and price")

print("Correlation between highway-mpg and price \n", df[['highway-mpg', 'price']].corr())   # -0.704

# see if "Peak-rpm" as a predictor variable of "price"
sns.regplot(x='peak-rpm', y='price', data=df)
plt.savefig("scatter plot of peak-rpm and price")
plt.title("Scatter plot of peak-rpm and price")

print("Correlation between peak-rpm and price \n", df[['peak-rpm', 'price']].corr())    #-0.10


# look at the boxplot between "body-style" and "price"
sns.boxplot(x="body-style", y="price", data=df)
plt.savefig("boxplot_bodystyle_price")
plt.title("Boxplot_between bodystyle_and price")

# look at the boxplot between "engine-location" and "price"
sns.boxplot(x="engine-location", y="price", data=df)
plt.title("Boxplot between engine-location and price")
plt.savefig("boxplot_engineLocation_price")

# look at the boxplot between "drive-wheels" and "price"
sns.boxplot(x="drive-wheels", y="price", data=df)
plt.savefig("boxplot_drive-wheels_price")
plt.title("Boxplot_between drive-wheels_and price")

# statistical summary
print(df.describe(include="all"))

# apply the "value_counts" method on the column 'drive-wheels'.
print("Value counts of drive-wheels \n", df["drive-wheels"].value_counts())

# convert the series to a Dataframe

print(df['drive-wheels'].value_counts().to_frame())

# save the results to the dataframe "drive_wheels_counts" and rename the column 'drive-wheels' to 'value_counts'

drive_wheels_counts = df["drive-wheels"].value_counts().to_frame()
drive_wheels_counts.rename(columns={"drive-wheels": "value_counts"}, inplace=True)

print(drive_wheels_counts)


# rename the index to 'drive-wheels'
drive_wheels_counts.index.name = "drive-wheels"
print(drive_wheels_counts)

# group by the variable "drive-wheels" and display the mean proce
print("grouping \n", df[["price", "drive-wheels"]].groupby(["drive-wheels"], as_index=False).mean())

# group by both 'drive-wheels' and 'body-style'
df_grouped = df[["drive-wheels", "body-style", "price"]].groupby(['drive-wheels', 'body-style'], as_index=False).mean()
print("grouping \n", df_grouped)

# make grouping into a pivot table
df_grouped_pivoted = pd.pivot(index="drive-wheels", columns="body-style", data=df_grouped)
print("after pivoting ", df_grouped_pivoted)

# fill missing values with 0
df_grouped_pivoted=df_grouped_pivoted.fillna(0)

print("after pivoting and filling missing values with 0s \n", df_grouped_pivoted)

# use a heat map to visualize the relationship between Body Style vs Price
plt.pcolor(df_grouped_pivoted, cmap='RdBu')
plt.colorbar()
plt.savefig("heatmap")

#plt.show()












