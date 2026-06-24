# ==========================
# HOUSE RENT DATA ANALYSIS
# ==========================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================
# LOAD DATASET
# ==========================

# Upload your CSV file in Colab and update the filename if needed
df = pd.read_csv('/content/drive/MyDrive/House_Rent_Dataset.csv')

# ==========================
# DATASET OVERVIEW
# ==========================

print("========== DATASET OVERVIEW ==========")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

# ==========================
# CHECK MISSING VALUES
# ==========================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Handle Missing Values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].mean())

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# ==========================
# DATA CLEANING
# ==========================

print("\n========== DUPLICATES ==========")
print("Number of Duplicates:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# Convert Date Column
if 'Posted On' in df.columns:
    df['Posted On'] = pd.to_datetime(df['Posted On'])

# ==========================
# EXPLORATORY DATA ANALYSIS
# ==========================

numeric_cols = df.select_dtypes(include=np.number).columns

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(df[numeric_cols[0]], bins=30, kde=True)
plt.title(f'Distribution of {numeric_cols[0]}')
plt.show()

# Boxplot
plt.figure(figsize=(10,5))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot of Numeric Features")
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(),
            annot=True,
            cmap='coolwarm',
            fmt='.2f')

plt.title("Feature Correlation Matrix")
plt.show()

# ==========================
# STATISTICAL SUMMARY
# ==========================

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# ==========================
# LINEAR REGRESSION MODEL
# ==========================

if 'Rent' in df.columns:

    X = df[numeric_cols].drop('Rent', axis=1)
    y = df['Rent']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n========== MODEL PERFORMANCE ==========")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)

else:
    print("Rent column not found in dataset.")

print("\n========== ANALYSIS COMPLETED ==========")# House-Rent-Data-Analysis-and-Prediction
An end-to-end data analytics project that performs data cleaning, exploratory data analysis (EDA), visualization, statistical analysis, and rent price prediction using Linear Regression on a House Rent dataset.
