
# STUDENT PERFORMANCE ANALYSIS
# Complete Data Cleaning and Visualization Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CREATE SAMPLE DATASET
np.random.seed(42)

df = pd.DataFrame({
    'student_id': range(1, 121),
    'gender': np.random.choice(['Male', 'Female'], 120),
    'study_hours_per_day': np.random.randint(1, 10, 120),
    'attendance_percent': np.random.randint(50, 101, 120),
    'previous_score': np.random.randint(30, 101, 120),
    'sleep_hours': np.random.randint(4, 10, 120),
    'final_score': np.random.randint(30, 101, 120)
})

# Add missing values, duplicates and outliers
df.loc[5, 'study_hours_per_day'] = np.nan
df.loc[10, 'attendance_percent'] = np.nan
df.loc[15, 'final_score'] = np.nan
df.loc[20, 'final_score'] = 150
df.loc[25, 'study_hours_per_day'] = 25
df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

# 2. DATA OVERVIEW
print("FIRST FIVE ROWS")
display(df.head())

print("DATASET INFORMATION")
print(df.info())

print("STATISTICAL SUMMARY")
display(df.describe())

print("MISSING VALUES")
print(df.isnull().sum())

print("DUPLICATE ROWS:", df.duplicated().sum())

# 3. REMOVE DUPLICATES
df.drop_duplicates(inplace=True)

# 4. HANDLE INVALID VALUES
df.loc[df['study_hours_per_day'] > 16,
       'study_hours_per_day'] = np.nan

df.loc[~df['attendance_percent'].between(0, 100),
       'attendance_percent'] = np.nan

df.loc[~df['final_score'].between(0, 100),
       'final_score'] = np.nan

# 5. HANDLE MISSING VALUES
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

# 6. OUTLIER DETECTION USING IQR
print("\nOUTLIER DETECTION")

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(col, "Outliers:", len(outliers))

# 7. HANDLE OUTLIERS USING CAPPING
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

# 8. PERFORMANCE CATEGORIES
df['performance_level'] = pd.cut(
    df['final_score'],
    bins=[-1, 39, 59, 79, 100],
    labels=['Needs Improvement', 'Average', 'Good', 'Excellent']
)

# 9. DATA VISUALIZATION
sns.set_theme(style="whitegrid")

# Graph 1: Final Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['final_score'], bins=10, kde=True)
plt.title("Distribution of Student Final Scores")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.show()

# Graph 2: Study Hours vs Final Score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='study_hours_per_day',
                y='final_score', hue='gender')
plt.title("Study Hours vs Final Score")
plt.show()

# Graph 3: Gender-wise Average Score
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x='gender', y='final_score')
plt.title("Average Final Score by Gender")
plt.show()

# Graph 4: Correlation Heatmap
plt.figure(figsize=(9, 6))
sns.heatmap(df[numeric_cols].corr(),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Between Student Performance Factors")
plt.show()

# Graph 5: Performance Categories
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x='performance_level')
plt.title("Student Performance Categories")
plt.xticks(rotation=15)
plt.show()

# 10. FINAL ANALYSIS
print("\nFINAL PROJECT SUMMARY")
print("Total Students:", len(df))
print("Average Final Score:", round(df['final_score'].mean(), 2))
print("Highest Score:", df['final_score'].max())
print("Lowest Score:", df['final_score'].min())

pass_rate = (df['final_score'] >= 40).mean() * 100
print("Pass Percentage:", round(pass_rate, 2), "%")

print("\nPerformance Category Counts:")
print(df['performance_level'].value_counts())

# 11. SAVE CLEANED DATASET
df.to_csv("Student_Performance_Cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Download cleaned CSV in Google Colab
from google.colab import files
files.download("Student_Performance_Cleaned.csv")



