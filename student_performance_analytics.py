import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('student_data.csv')

# Display dataset
print("\nStudent Dataset:\n")
print(df)

# Convert Pass/Fail into numeric
df['FinalResult'] = df['FinalResult'].map({'Pass': 1, 'Fail': 0})

# Dataset Information
print("\nDataset Information:\n")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:\n")
print(df.describe())

# Average values
print("\nAverage Study Hours:", df['StudyHours'].mean())
print("Average Attendance:", df['Attendance'].mean())

# Visualization
plt.figure(figsize=(8,5))
plt.bar(df.index, df['PreviousMarks'])
plt.xlabel('Students')
plt.ylabel('Previous Marks')
plt.title('Student Previous Marks Analysis')
plt.show()

# Features and Target
X = df[['StudyHours', 'Attendance', 'PreviousMarks', 'Assignments']]
y = df['FinalResult']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Predict new student result
new_student = [[4, 80, 75, 78]]

result = model.predict(new_student)

if result[0] == 1:
    print("\nPrediction: Student is likely to PASS")
else:
    print("\nPrediction: Student is likely to FAIL")