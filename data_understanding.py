import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Loan_Default.csv")

# print("shape :\n",df.shape) # size of the dataset
# print("columns :\n",df.columns)
# print("info :\n")
# df.info(True)
# print("--------------------------------------------------------------------")
print("description:\n",df.describe(include="all").T)
# display(df.head())
# missing_vals_report = (df.isna().mean() * 100).sort_values(ascending=False)
# print("miss report : \n",missing_vals_report)


# b =df['Status'].value_counts(normalize=True)
# c = df['Status'].describe()
# d = df.duplicated().sum()
# print(d)
# df.hist(figsize=(15, 10))
# plt.show()