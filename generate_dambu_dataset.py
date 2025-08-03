
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import display
from sklearn.preprocessing import StandardScaler

"""
Generate synthetic dataset for Dambu glycemic response study based on Malimuna Ladii Aliyu et al. (2020).
The dataset includes nutritional composition (millet, maize, beans, groundnut, vegetables, oil), age, blood glucose responses,
IAUC, GI, and sensory attributes. Aligned with DSHub Internship Program (Cohort B, AI/ML Track) to predict GI from meal
composition. Assumptions: IAUC_Standard_Meal ~ N(6000, 500) based on 50g glucose; glucose stats and GI from paper Tables 1-4.
"""

# Set random seed for reproducibility
np.random.seed(2025)

# Configuration
NUM_SUBJECTS = 100  # Total subjects (50 male, 50 female)
IAUC_STANDARD_MEAL_MEAN = 6000  # Mean IAUC for 50g glucose (mmol/L*min)
IAUC_STANDARD_MEAL_STD = 500    # Std dev for standard meal IAUC
TIME_POINTS = [0, 30, 60, 90, 120]  # Measurement time points (minutes)
AGE_RANGE = (18, 65)  # Age range for subjects

# Validate TIME_POINTS
if any(TIME_POINTS[i] < 0 or TIME_POINTS[i] <= TIME_POINTS[i-1] for i in range(1, len(TIME_POINTS))):
    raise ValueError("TIME_POINTS must be positive and strictly increasing.")

# Nutritional composition (approximate weighted averages based on ingredients)
INGREDIENT_PROPORTIONS = {'Maize': 0.4, 'Millet': 0.4, 'Beans': 0.1, 'Groundnut': 0.05, 'Veg_Oil': 0.05}
NUTRITIONAL_COMPOSITION = {
    'Maize': {'Carb_Percent': 73.0, 'Protein_Percent': 9.4, 'Fat_Percent': 4.7, 'Fiber_Percent': 7.3},
    'Millet': {'Carb_Percent': 72.8, 'Protein_Percent': 11.0, 'Fat_Percent': 4.2, 'Fiber_Percent': 8.5},
    'Beans': {'Carb_Percent': 20.0, 'Protein_Percent': 20.0, 'Fat_Percent': 1.0, 'Fiber_Percent': 15.0},
    'Groundnut': {'Carb_Percent': 15.0, 'Protein_Percent': 25.0, 'Fat_Percent': 50.0, 'Fiber_Percent': 8.0},
    'Veg_Oil': {'Carb_Percent': 0.0, 'Protein_Percent': 0.0, 'Fat_Percent': 100.0, 'Fiber_Percent': 0.0}
}

# Glucose stats (aligned with Tables 1, 2, and 3; time-series approximated)
GLUCOSE_BASE_STATS = {
    'Maize': {'Male': {'Fasting': (4.82, 0.64)}, 'Female': {'Fasting': (5.2, 0.8)}},
    'Millet': {'Male': {'Fasting': (4.94, 0.9)}, 'Female': {'Fasting': (5.14, 0.53)}}
}
TIME_FACTORS = {'30min': 1.2, '60min': 1.1, '90min': 1.0, '120min': 0.95}  # Relative to fasting

# GI targets (from Table 4)
GI_TARGETS = {'Maize': {'Male': 40.33, 'Female': 41.51}, 'Millet': {'Male': 44.83, 'Female': 44.31}}

# Sensory attributes (from Table 5, approximate means and std devs)
SENSORY_ATTRIBUTES = {
    'Maize': {
        'Flavor': (7.35, 1.09), 'Taste': (7.40, 1.12), 'Texture': (6.90, 1.12),
        'Consistency': (6.75, 1.07), 'Color': (7.30, 0.73), 'Overall_Acceptability': (7.70, 0.80)
    },
    'Millet': {
        'Flavor': (7.15, 0.88), 'Taste': (7.10, 1.12), 'Texture': (6.65, 0.75),
        'Consistency': (6.90, 1.12), 'Color': (7.35, 0.81), 'Overall_Acceptability': (7.50, 1.00)
    }
}

def calculate_iauc(glucose_values, time_points):
    """
    Calculate Incremental Area Under the Curve (IAUC) using the trapezoidal rule.
    Only positive incremental areas above baseline (fasting glucose) are summed.

    Args:
        glucose_values (list): Blood glucose values (mmol/L) at each time point.
        time_points (list): Time points (minutes) corresponding to glucose values.

    Returns:
        float: Calculated IAUC (mmol/L*min).

    Raises:
        ValueError: If glucose_values and time_points have different lengths.
    """
    if len(glucose_values) != len(time_points):
        raise ValueError("Glucose values and time points must have equal length.")
    
    baseline = glucose_values[0]
    incremental_values = [max(0, g - baseline) for g in glucose_values]
    iauc = np.trapezoid(incremental_values, time_points)  # Replaced np.trapz with np.trapezoid
    return max(0, iauc)

# Generate synthetic data
data = []
for i in range(1, NUM_SUBJECTS + 1):
    subject_id = f'S{i:03d}'
    gender = 'Male' if i <= NUM_SUBJECTS // 2 else 'Female'
    age = round(np.random.uniform(AGE_RANGE[0], AGE_RANGE[1]), 1)
    portion_size_g = round(np.random.uniform(50, 150), 1)
    boiled = np.random.randint(0, 2)  # 0 = roasted, 1 = boiled

    for dambu_type in ['Maize', 'Millet']:
        row = {
            'Subject_ID': subject_id,
            'Gender': gender,
            'Age': age,
            'Dambu_Type': dambu_type,
            'Portion_Size_g': portion_size_g,
            'Boiled': boiled
        }
        # Calculate weighted nutritional composition with variability
        for nutrient in ['Carb_Percent', 'Protein_Percent', 'Fat_Percent', 'Fiber_Percent']:
            base_value = sum(INGREDIENT_PROPORTIONS[ing] * NUTRITIONAL_COMPOSITION[ing][nutrient]
                           for ing in INGREDIENT_PROPORTIONS)
            row[nutrient] = round(max(0, np.random.normal(base_value, base_value * 0.05)), 1)
        
        # Generate glucose values with time-series based on fasting and factors
        glucose_values = []
        fasting_mean, fasting_std = GLUCOSE_BASE_STATS[dambu_type][gender]['Fasting']
        fasting_glucose = max(0.1, np.random.normal(fasting_mean, fasting_std))
        row['Glucose_Fasting_mmol_L'] = round(fasting_glucose, 2)  # Add fasting value to row
        glucose_values.append(fasting_glucose)
        for time_label, factor in TIME_FACTORS.items():
            glucose_val = max(0.1, fasting_glucose * factor + np.random.normal(0, 0.1))
            row[f'Glucose_{time_label}_mmol_L'] = round(glucose_val, 2)
            glucose_values.append(glucose_val)
        
        # Ensure realistic glucose curve
        if max(glucose_values[1:]) <= glucose_values[0]:
            glucose_values[1] = max(glucose_values[1], glucose_values[0] * 1.2)
            row['Glucose_30min_mmol_L'] = round(glucose_values[1], 2)
        if glucose_values[4] > glucose_values[0] + 0.2:
            glucose_values[4] = glucose_values[0] + 0.1
            row['Glucose_120min_mmol_L'] = round(glucose_values[4], 2)

        # Calculate IAUC and GI
        row['IAUC_Test_Meal'] = round(calculate_iauc(glucose_values, TIME_POINTS), 2)
        row['IAUC_Standard_Meal'] = round(max(1000, np.random.normal(IAUC_STANDARD_MEAL_MEAN, IAUC_STANDARD_MEAL_STD)), 2)
        if row['IAUC_Standard_Meal'] == 0:
            raise ValueError(f"Invalid IAUC_Standard_Meal for subject {subject_id}: {row['IAUC_Standard_Meal']}")
        target_gi = GI_TARGETS[dambu_type][gender]
        row['Calculated_Glycemic_Index_Percent'] = round(max(0, min(100, target_gi + np.random.normal(0, 1))), 2)

        # Add sensory attributes
        for attr, (mean, std) in SENSORY_ATTRIBUTES[dambu_type].items():
            row[attr] = round(max(1, min(9, np.random.normal(mean, std))), 2)

        data.append(row)

# Create DataFrame
df = pd.DataFrame(data)
ordered_columns = [
    'Subject_ID', 'Gender', 'Age', 'Dambu_Type', 'Portion_Size_g', 'Boiled',
    'Carb_Percent', 'Protein_Percent', 'Fat_Percent', 'Fiber_Percent',
    'Glucose_Fasting_mmol_L', 'Glucose_30min_mmol_L', 'Glucose_60min_mmol_L',
    'Glucose_90min_mmol_L', 'Glucose_120min_mmol_L', 'IAUC_Test_Meal',
    'IAUC_Standard_Meal', 'Calculated_Glycemic_Index_Percent', 'Flavor',
    'Taste', 'Texture', 'Consistency', 'Color', 'Overall_Acceptability'
]
if set(df.columns) != set(ordered_columns):
    raise ValueError(f"DataFrame columns {set(df.columns)} do not match expected {set(ordered_columns)}")
df = df[ordered_columns]

# Check for empty DataFrame
if df.empty:
    raise ValueError("Generated DataFrame is empty")

# Save to CSV with overwrite warning
if os.path.exists('dambu_dataset.csv'):
    print("Warning: 'dambu_dataset.csv' already exists.")
df.to_csv('dambu_dataset.csv', index=False)


# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)


# Visualize glucose response trends
# plt.figure(figsize=(10, 6))
# time_labels = ['Fasting', '30min', '60min', '90min', '120min']
# for t in time_labels:
#     if f'Glucose_{t}_mmol_L' not in df.columns:
#         raise ValueError(f"Column Glucose_{t}_mmol_L not found in DataFrame")
# for dambu_type in ['Maize', 'Millet']:
#     subset = df[df['Dambu_Type'] == dambu_type]
#     if subset.empty:
#         print(f"Warning: No data for Dambu_Type {dambu_type}")
#         continue
#     means = [subset[f'Glucose_{t}_mmol_L'].mean() for t in time_labels]
#     plt.plot(TIME_POINTS, means, marker='o', label=f'{dambu_type}')
# plt.xlabel('Time (minutes)')
# plt.ylabel('Blood Glucose (mmol/L)')
# plt.title('Average Glucose Response for Dambu (Maize vs. Millet)')
# plt.legend()
# plt.grid(True)
# if os.path.exists('glucose_response_trends.png'):
#     print("Warning: 'glucose_response_trends.png' already exists.")
# plt.savefig('glucose_response_trends.png')
# plt.show()

# Print sample data
print("Generated Mock Data (First 5 Rows):")
display(df.head())