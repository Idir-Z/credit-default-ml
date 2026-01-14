from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("Loan_Default.csv")

profile = ProfileReport(df, title="Dataset Report", explorative=True)
profile.to_file("docs/data_report.html")