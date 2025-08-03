import pandas as pd
import matplotlib.pyplot as plt

# Loading your dataset
df = pd.read_csv("dambu_dataset.csv")

# Grouping it by Dambu_Type and compute mean glucose response at each time point
glucose_columns = [
    "Glucose_Fasting_mmol_L",
    "Glucose_30min_mmol_L",
    "Glucose_60min_mmol_L",
    "Glucose_90min_mmol_L",
    "Glucose_120min_mmol_L"
]

average_glucose = df.groupby("Dambu_Type")[glucose_columns].mean()

# Plot
time_points = [0, 30, 60, 90, 120]
plt.figure(figsize=(10, 6))

for dambu_type in average_glucose.index:
    plt.plot(time_points, average_glucose.loc[dambu_type], label=dambu_type, marker='o')

plt.title("Glucose Response Trends for Maize vs Millet Dambu")
plt.xlabel("Time (minutes)")
plt.ylabel("Blood Glucose (mmol/L)")
plt.legend(title="Dambu Type")
plt.grid(True)
plt.tight_layout()
plt.show()
