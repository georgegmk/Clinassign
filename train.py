import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF

# Load dataset
df = pd.read_csv('extracted_features.csv')

# Drop unnecessary columns
df = df.drop(columns=['Gender', 'Age'])

# Select only the specified columns for feature sum
selected_columns = [
    'Medications Administered',
    'Procedures Performed',
    'Pain Management Strategies',
    'Identification of Diagnoses'
]

# Calculate feature sum for selected columns only
feature_sums = df[selected_columns].sum(axis=1)

# Define grading logic based on new thresholds
def assign_grade(sum_value):
    if sum_value > 15:  # Higher feature counts
        return 'O'
    elif 8 <= sum_value <= 15:  # Medium feature counts
        return 'A'
    else:  # Lower feature counts
        return 'B'

# Generate grades based on feature sums
predicted_grades = feature_sums.apply(assign_grade)

# Create DataFrame with predictions and save to CSV
results_df = pd.DataFrame({
    'RF_Model_Grade': predicted_grades
})

# Save to CSV
results_df.to_csv('rf_model_predictions.csv', index=False)
print("\nPredictions have been saved to 'rf_model_predictions.csv'")

from google.colab import files
files.download("rf_model_predictions.csv")

