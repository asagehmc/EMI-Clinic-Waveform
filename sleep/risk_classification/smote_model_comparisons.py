import numpy as np
import matplotlib.pyplot as plt

from models import score_summary_model, score_summary_model_with_smote
from preprocessing import load_preprocessing_data

# Loading the dataset
X_sum, y_sum, X_sum_dem, y_sum_dem, X_ts, y_ts = load_preprocessing_data()

# Defining the metric names corresponding to the tuple returned by the scoring functions:
# (direct_accuracy, risk_accuracy, no_risk_accuracy, FPR, FNR)
metric_names = [
    'Direct Accuracy',    # overall % of patients correctly labeled
    'Risk Accuracy',      # accuracy of patients who are at risk
    'No-Risk Accuracy',   # accuracy of patients who are not at risk
    'False Positive Rate',# % of patients predicted as at risk when they are actually not at risk
    'False Negative Rate' # % of patients predicted as not at risk when they are actually at risk
]

# Setting the number of runs to average results over (to account for random splitting)
num_runs = 100

# Defining the list of models to evaluate
model_list = ['svc', 'rfc', 'knc']

# Creating dictionaries to store the averaged metrics for each model
metrics_no_smote = {}
metrics_with_smote = {}

# Looping over each model and collect the metrics before and after SMOTE
for model in model_list:
    results_no = []
    results_yes = []
    for _ in range(num_runs):
        # Evaluating the model without SMOTE (returns a tuple of 5 values)
        score_no = score_summary_model(X_sum, y_sum, model)
        # Evaluating the model with SMOTE using the specified oversampling method ("smote")
        score_yes = score_summary_model_with_smote(X_sum, y_sum, model, test_size=0.2, oversampling_method="smote")
        results_no.append(score_no)
        results_yes.append(score_yes)
    # Averaging the metrics across the runs (axis=0 averages each metric separately)
    avg_no = np.mean(results_no, axis=0)
    avg_yes = np.mean(results_yes, axis=0)
    metrics_no_smote[model] = avg_no
    metrics_with_smote[model] = avg_yes

# Creating one subplot per model
num_models = len(model_list)
fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 6), sharey=True)

# If only one model exists, ensuring that the axes are iterable
if num_models == 1:
    axes = [axes]

# x-axis positions for the metric groups
x = np.arange(len(metric_names))
width = 0.35  # the width of the bars

# Looping over each model to create its grouped bar chart
for i, model in enumerate(model_list):
    ax = axes[i]
    # Getting the averaged metrics for the current model
    values_no = metrics_no_smote[model]
    values_yes = metrics_with_smote[model]
    # Creating the bars: one group for metrics before SMOTE and one for after SMOTE
    bars_no = ax.bar(x - width / 2, values_no, width, label='Before SMOTE')
    bars_yes = ax.bar(x + width / 2, values_yes, width, label='After SMOTE')
    # Set the x-axis labels and title for this subplot
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.set_title(model.upper())
    ax.set_ylabel('Metric Value (%)')  # assuming the metrics are reported in percentage
    ax.legend()

    # Annotating each bar with its height value for clarity
    for bar in bars_no:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # vertical offset for the text
                    textcoords="offset points",
                    ha='center', va='bottom')
    for bar in bars_yes:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # vertical offset for the text
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()