import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data obtained from report
data = {
    'Class': ['cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi'],
    'Precision': [0.9017, 0.9479, 0.9915, 0.9980, 0.9648],
    'Recall': [0.9720, 0.9100, 0.9320, 0.9960, 0.9880],
    'F1-Score': [0.9355, 0.9286, 0.9608, 0.9970, 0.9763]
}

# Create dataframe
df = pd.DataFrame(data)

# Transform dataframe
df_melted = df.melt(id_vars='Class', var_name='Metric', value_name='Score')

# Create graph
plt.figure(figsize=(10, 6))
sns.barplot(x='Class', y='Score', hue='Metric', data=df_melted, palette='viridis')
plt.title('Classification Report Metrics')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.savefig('../results/figs/classification_report.png')
plt.show()
