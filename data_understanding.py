import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Loan_Default.csv")

# print("shape :\n",df.shape) # size of the dataset
# print("columns :\n",df.columns)
# print("info :\n")
# df.info(True)
# print("--------------------------------------------------------------------")
# print("description:\n",df.describe(include="all").T)

# missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
# print("miss report : \n",missing_vals_report)


# missing_pct[missing_pct > 0].plot(kind='bar')
# plt.title("Missing Values Percentage")
# plt.show()


# class_distr =df['Status'].value_counts() # see the distribution of the target column values to see whether we have an inbalance
# # print(class_distr)
# class_distr.plot(kind='bar')
# plt.title("Target Class Distribution")
# plt.savefig("figures/loan_status_distribution.png")
# plt.show()

# c = df['Status'].describe() # this is used if our problem was a regression
# print(c)

# d = df.duplicated().sum()


# here we separate the categorical and numerical cols
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

categorical_cols = df.select_dtypes(include=['object']).columns
# we remove useless cols like year which is constant(2019) and ID which has unique values
numeric_cols = numeric_cols.drop(['ID', 'year'])
