import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("dambu_dataset.csv")

# Dropping the non-numeric or irrelevant columns
df = df.drop(columns=["Subject_ID", "Gender", "Dambu_Type", "Flavor", "Taste", "Texture", "Consistency", "Color", "Overall_Acceptability"])

# Defining my features and target
X = df[[
    "Age", "Portion_Size_g", "Boiled",
    "Carb_Percent", "Protein_Percent",
    "Fat_Percent", "Fiber_Percent"
]]
y = df["Calculated_Glycemic_Index_Percent"]

# Scaling the features from sklearn
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting my dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Training my regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Plotting the model's predictions vs actual ones

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Glycemic Index")
plt.ylabel("Predicted Glycemic Index")
plt.title("Actual vs Predicted GI")
plt.grid(True)
plt.show()
