import pandas as pd

# Load the high-security report
df = pd.read_csv(r'data\attendance_report.csv')

# 1. Generate a Simplified Attendance Sheet (What the faculty actually wants to see)
summary = df[['Name', 'Date', 'Time', 'Status']].drop_duplicates(subset=['Name', 'Date'])
summary.to_csv(r'data\faculty_submission.csv', index=False)

# 2. Print Project Statistics for the Viva
print("--- Project Analytics ---")
print(f"Total Records Captured: {len(df)}")
print(f"Unique Students Verified: {df['Name'].nunique()}")
print(f"Avg Biometric Quality (Symmetry): {df['Symmetry'].mean():.2f}")

print("\nSuccess! 'faculty_submission.csv' created for your records.")