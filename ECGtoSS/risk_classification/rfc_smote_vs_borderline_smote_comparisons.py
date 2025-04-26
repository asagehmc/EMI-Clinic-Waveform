import numpy as np
import matplotlib.pyplot as plt

from models import score_summary_model_with_smote
from preprocessing import load_preprocessing_data

# Loading the dataset
X_sum, y_sum, X_sum_dem, y_sum_dem, X_ts, y_ts = load_preprocessing_data()

# Define the five metric names (the order returned by your 'accuracies' function).
metric_names = ['Direct Accuracy', 'Risk Accuracy', 'No-Risk Accuracy', 'FPR', 'FNR']

num_runs = 10

# Creating dictionaries to store the averaged metrics for each of the oversampling methods
results = {'smote': [], 'borderline_smote': []}

for method in ['smote', 'borderline_smote']:
    run_metrics = []
    for i in range(num_runs):
        # For RFC, using score_summary_model_with_smote which returns (direct_accuracy, risk_accuracy, no_risk_accuracy, FPR, FNR)
        metrics = score_summary_model_with_smote(X_sum, y_sum, model_type="rfc", test_size=0.2, oversampling_method=method)
        run_metrics.append(metrics)
    # Averaging metrics over the runs
    results[method] = np.mean(np.array(run_metrics), axis=0)

# Printing averaged results for verification.
print("Averaged Metrics for RFC with SMOTE:")
for name, val in zip(metric_names, results['smote']):
    print(f"{name}: {val:.2f}")

print("\nAveraged Metrics for RFC with Borderline-SMOTE:")
for name, val in zip(metric_names, results['borderline_smote']):
    print(f"{name}: {val:.2f}")

# Creating a grouped bar chart comparing RFC performance with regular SMOTE vs. Borderline-SMOTE
x = np.arange(len(metric_names))  # one group for each metric
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(10, 6))
bars_smote = ax.bar(x - width/2, results['smote'], width, label='SMOTE')
bars_borderline = ax.bar(x + width/2, results['borderline_smote'], width, label='Borderline-SMOTE')

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('Metric Value (%)')
ax.set_title('RFC Performance Metrics: SMOTE vs. Borderline-SMOTE')
ax.set_xticks(x)
ax.set_xticklabels(metric_names, rotation=45, ha='right')
ax.legend()

# Annotating each bar with its numeric value
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),  # 3-point vertical offset.
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bars_smote)
autolabel(bars_borderline)

plt.tight_layout()
plt.show()