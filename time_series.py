import pandas as pd
import numpy as np 

df = pd.read_csv('LoanStats3c_update.csv', index_col=0, low_memory=False)

# converts string to datetime object in pandas:
df['issue_d_format'] = pd.to_datetime(df['issue_d']) 
dfts = df.set_index('issue_d_format') 
year_month_summary = dfts.groupby(lambda x : x.year * 100 + x.month).count()
loan_count_summary = year_month_summary['issue_d']

import matplotlib.pyplot as plt

plt.figure()
year_month_summary.hist(column='issue_d')
plt.show()
import statsmodels.api

plt.figure()
statsmodels.api.graphics.tsa.plot_acf(loan_count_summary)
plt.show()

plt.figure()
statsmodels.api.graphics.tsa.plot_pacf(loan_count_summary)
plt.show()