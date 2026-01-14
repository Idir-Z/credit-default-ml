import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



df = pd.read_csv("Loan_Default.csv")

# here we separate the categorical and numerical cols
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

categorical_cols = df.select_dtypes(include=['object']).columns

# we get the descriptions for the num and categ columns
num_desc = df[numeric_cols].describe().T
cat_summary = {
    col: {
        "count": df[col].count(),
        "unique": df[col].nunique(),
        "missing": df[col].isna().sum(),
        "most_frequent": df[col].value_counts().idxmax(),
        "most_frequent_freq": df[col].value_counts().max()
    }
    for col in categorical_cols
}
cat_summary_df = pd.DataFrame(cat_summary).T

# here we identify constant columns
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
# print("Constant columns:", constant_cols)

# identify identifier-like cols
id_cols = [col for col in df.columns if df[col].nunique() == len(df)]
# print("Identifier columns:", id_cols)

# then we drop the useless cols
numeric_cols = numeric_cols.drop(['ID', 'year'])
df = df.drop(columns=constant_cols + id_cols)

# missing data handling (leakage-safe)
# we split the set into train and test beforehand in order to prevent data-leakage
X = df.drop(columns=['Status'])
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# missing numerical data report
# missing_pct = (X_train.isna().mean() * 100).sort_values(ascending=False)
num_missing_pct =  (X_train[numeric_cols].isna().mean() * 100).sort_values(ascending=False)
cat_missing_pct =  (X_train[categorical_cols].isna().mean() * 100).sort_values(ascending=False)

# visualisation of missing data
# missing_pct[missing_pct > 0].plot(kind='bar', figsize=(10,4))
# plt.title("Missing Values Percentage")
# plt.ylabel("%")
# plt.savefig("figures/missing_values_report.png")
# plt.show()





import matplotlib.pyplot as plt

# --------------------------------------------------
# Target class distribution analysis (training set)
# --------------------------------------------------

# Compute class counts and proportions
class_counts = y_train.value_counts()
class_ratios = y_train.value_counts(normalize=True)

# Print numerical summary
# print("Target class counts (training set):")
# print(class_counts)

# print("\nTarget class proportions (training set):")
class_ratios =  (class_ratios * 100).round(2)
# print(class_ratios)

# --------------------------------------------------
# Baseline accuracy
# --------------------------------------------------
# Baseline accuracy corresponds to always predicting the majority class
baseline_accuracy = class_ratios.max()
# print(f"\nBaseline accuracy: {baseline_accuracy:.2%}")

# --------------------------------------------------
# Visualization of class distribution
# --------------------------------------------------
class_ratios.plot(kind='bar')
plt.title("Target Class Distribution (Training Set)")
plt.ylabel("Percentage")
plt.tight_layout()
# plt.savefig("figures/loan_status_distribution.png")
# plt.show()


# numerical features skewness
skewness = X_train[numeric_cols].skew().sort_values(ascending=False)
# print("skewness of vars: ",skewness)
# separate the highly skewed variables
highly_skewed = skewness[abs(skewness) > 1]
# and then we visualize them
for col in highly_skewed.index:
    sns.histplot(X_train[col], bins=50, kde=True)
    plt.title(f"Distribution of {col}")
    # plt.savefig("figures/"+col+"_distribution.png")
    # plt.show()
# now we get into outliers
quantiles = X_train[numeric_cols].quantile([0.01, 0.99])
outlier_summary = {}

for col in numeric_cols:
    lower = X_train[col].quantile(0.01)
    upper = X_train[col].quantile(0.99)
    outlier_summary[col] = {
        "below_1%": (X_train[col] < lower).mean() * 100,
        "above_99%": (X_train[col] > upper).mean() * 100
    }

# print("outlier summary: ",outlier_summary)

# now we get the correclation matrix to determine redundant variables
corr = X_train[numeric_cols].corr()

# print("correlation matrix: ",corr)

# Categorical variable analysis
unique_counts = X_train[categorical_cols].nunique().sort_values(ascending=False)

# print("categorical unique counts: ",unique_counts)

# percentage distribution for each category
rows = []

for col in categorical_cols:
    value_counts = X_train[col].value_counts(normalize=True)
    for category, proportion in value_counts.items():
        rows.append({
            "feature": col,
            "category": category,
            "percentage": proportion * 100
        })

category_distribution_report = pd.DataFrame(rows)
# identifying rare categories
rare_categories = {}

for col in categorical_cols:
    freq = X_train[col].value_counts(normalize=True)
    rare = freq[freq < 0.01].index.tolist()
    if rare:
        rare_categories[col] = rare

rare_categories


with open("docs/dataset_description.md", "w") as f:
    f.write("## Numerical Features\n")
    f.write(num_desc.to_markdown())
    f.write("\n\n## Categorical Features\n")
    f.write(cat_summary_df.to_markdown())
    f.write("\n\n## Missing numerical values Report\n")
    f.write(num_missing_pct.to_markdown())
    f.write("\n\n## Missing categorical values Report\n")
    f.write(cat_missing_pct.to_markdown())
    f.write("\n\n## Skewness report\n")
    f.write(skewness.to_markdown())
    f.write("\n\n## Correlation matrix\n")
    f.write(corr.to_markdown())
    f.write("\n\n## Categorical features unique values\n")
    f.write(unique_counts.to_markdown())
    f.write("\n\n## Category distribution report\n")
    f.write(category_distribution_report.to_markdown(index=False))
