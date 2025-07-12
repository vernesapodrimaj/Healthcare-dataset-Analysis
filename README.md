# 🏥 Healthcare Dataset Analysis with Python

Exploratory Data Analysis (EDA) of a hospital dataset covering demographics, medical conditions, billing, and hospital stay details. The goal is to uncover patterns in treatment costs, conditions, gender distribution, and length of stay.

---

## 📁 Dataset Overview

- 📊 **Source**: [Kaggle Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
- 🔢 **Rows**: ~55,000
- 📌 **Columns**: Name, Age, Gender, Blood Type, Medical Condition, Doctor, Insurance, Admission/Discharge Dates, Billing, etc.

---

## 🧰 Tools Used

- `pandas` – data loading & cleaning  
- `matplotlib`, `seaborn` – visualization  
- `numpy` – basic math operations  
- `Jupyter Notebook` – interactive analysis

---

## ✅ Key Steps in Analysis

```python
# Load and inspect
df = pd.read_csv('healthcare_dataset.csv')
df.head()
df.info()
df.isnull().sum()

# Clean & transform
df['Billing Amount'] = df['Billing Amount'].round(2)
df['Name'] = df['Name'].astype(str).str.strip().str.title()
df = df.drop('Room Number', axis=1)

# Convert dates & calculate stay
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
df['Date of Discharge'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
df['Length of Stay'] = (df['Date of Discharge'] - df['Date of Admission']).dt.days

