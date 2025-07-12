# Healthcare Dataset Analysis with Python

Exploratory Data Analysis (EDA) of a hospital dataset covering demographics, medical conditions, billing, and hospital stay details. The goal is to uncover patterns in treatment costs, conditions, gender distribution, and length of stay.

---

## Dataset Overview

-  **Source**: [Kaggle Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
-  **Rows**: ~55,000
- **Columns**: Name, Age, Gender, Blood Type, Medical Condition, Doctor, Insurance, Admission/Discharge Dates, Billing, etc.

---

## Tools Used

- `pandas`: data loading & cleaning  
- `matplotlib`, `seaborn`: visualization  
- `numpy`: basic math operations  
- `Jupyter Notebook`: interactive analysis

---

## Key Steps in Analysis
```python
# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#  Load dataset
df = pd.read_csv('/Users/vernesapodrimaj/Downloads/healthcare_dataset.csv')

# Initial inspection
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.head())
print(df.columns)

# Clean and transform
df['Billing Amount'] = df['Billing Amount'].round(2)
df['Name'] = df['Name'].astype(str).str.strip().str.title()
df = df.drop('Room Number', axis=1)

# Convert date columns and calculate length of stay
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
df['Date of Discharge'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
df['Length of Stay'] = (df['Date of Discharge'] - df['Date of Admission']).dt.days

# Extract year and month for optional trend analysis
df['admission_year'] = df['Date of Admission'].dt.year
df['admission_month'] = df['Date of Admission'].dt.month_name()

# Average billing & length of stay per medical condition
print(df.groupby('Medical Condition')[['Billing Amount', 'Length of Stay']].mean().round(2).sort_values('Billing Amount'))

# Boxplot: Billing by medical condition
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Medical Condition', y='Billing Amount')
plt.title('Billing Amount by Medical Condition')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Boxplot: Length of stay by condition
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Medical Condition', y='Length of Stay')
plt.title('Length of Stay by Medical Condition')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation between numeric features
numeric_cols = df[['Age', 'Billing Amount', 'Length of Stay']]
plt.figure(figsize=(8, 5))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix: Numeric Features")
plt.tight_layout()
plt.show()

# Blood Type vs Medical Condition (count)
blood_condition_ct = pd.crosstab(df['Blood Type'], df['Medical Condition'])
plt.figure(figsize=(10, 6))
sns.heatmap(blood_condition_ct, cmap='Blues', annot=True, fmt='g')
plt.title('Blood Type vs Medical Condition (Patient Count)')
plt.xlabel('Medical Condition')
plt.ylabel('Blood Type')
plt.tight_layout()
plt.show()

# Avg billing by Blood Type & Medical Condition
avg_bill = df.groupby(['Blood Type', 'Medical Condition'])['Billing Amount'].mean().unstack().round(1)
plt.figure(figsize=(10, 6))
sns.heatmap(avg_bill, cmap='YlGnBu', annot=True)
plt.title('Avg Billing by Blood Type & Medical Condition')
plt.tight_layout()
plt.show()

# Gender vs Medical Condition (count)
gender_condition_ct = pd.crosstab(df['Gender'], df['Medical Condition'])
plt.figure(figsize=(10, 5))
sns.heatmap(gender_condition_ct, cmap='Purples', annot=True, fmt='g')
plt.title('Gender vs Medical Condition (Patient Count)')
plt.tight_layout()
plt.show()

# Avg billing by Gender & Blood Type
avg_gender_bt = df.groupby(['Gender', 'Blood Type'])['Billing Amount'].mean().unstack().round(1)
plt.figure(figsize=(8, 5))
sns.heatmap(avg_gender_bt, cmap='coolwarm', annot=True)
plt.title('Avg Billing by Gender & Blood Type')
plt.tight_layout()
plt.show()

# Medical Condition vs Admission Type (count)
condition_admission_ct = pd.crosstab(df['Medical Condition'], df['Admission Type'])
plt.figure(figsize=(10, 6))
sns.heatmap(condition_admission_ct, cmap='Greens', annot=True, fmt='g')
plt.title('Medical Condition vs Admission Type')
plt.xlabel('Admission Type')
plt.ylabel('Medical Condition')
plt.tight_layout()
plt.show()

#
df.to_csv("/Users/vernesapodrimaj/Documents/health_care.csv", index=False)
