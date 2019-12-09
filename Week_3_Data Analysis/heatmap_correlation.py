import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

path = "automobileEDA.csv"

df = pd.read_csv(path)

print(df.head)                   # [201 rows x 29 columns]

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

# group by the variable "drive-wheels" and display the mean price
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

# use a heat map to visualize the relationship between drive-wheel, body-style and price
plt.pcolor(df_grouped_pivoted, cmap='RdBu')
plt.colorbar()
plt.savefig("heatmap")


# find the Pearson correlation (default computation method if corr() function)
print("correlation \n", df.corr())

# calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'
correlation_coefficient, p_value = stats.pearsonr(df['wheel-base'], df['price'])

print("Correlation coefficient and p_value,     ", correlation_coefficient, p_value)

# calculate the Pearson Correlation Coefficient and P-value of 'horsepower' and 'price'
corr_coeff, pValue = stats.pearsonr(df["horsepower"], df["price"])
print("Correlation coefficient and p_value,     ", corr_coeff, pValue)

# ANOVA
# see if different types 'drive-wheels' impact 'price'

df_anova = df[["drive-wheels", "body-style", "price"]]
#print("df_anova \n", df_anova)
df_anova_grouped = df_anova[["drive-wheels", "price"]].groupby(["drive-wheels"])
#grouped_test2 = df_anova[['drive-wheels', 'price']].groupby(['drive-wheels'])
print("df grouped by drive-wheels \n", df_anova_grouped.head(2))

f_Value, p_Value = stats.f_oneway(df_anova_grouped.get_group("fwd")["price"], df_anova_grouped.get_group("rwd")["price"], df_anova_grouped.get_group("4wd")["price"])

#f_Value, p_Value = stats.f_oneway(df_anova_grouped.get_group('fwd')['price'], df_anova_grouped.get_group('rwd')['price'], df_anova_grouped.get_group('4wd')['price'])

print("ANOVA Statistics: ", f_Value, p_Value)











