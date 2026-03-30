# student_advisor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Creating a sample dataset (simulating real students)
# --------------------------------------------------

np.random.seed(42)   # keeping results consistent

total_students = 250

# generating realistic ranges
hours_studied = np.random.uniform(1, 10, total_students)
attendance_percent = np.random.uniform(60, 100, total_students)
quiz_marks = np.random.uniform(40, 100, total_students)

# --------------------------------------------------
# Deciding whether a student passes or not
# (simple weighted logic based on habits)
# --------------------------------------------------

final_score = (
    0.5 * hours_studied +
    0.2 * attendance_percent +
    0.3 * quiz_marks
)

# threshold chosen after some manual observation
results = []
for score in final_score:
    if score > 45:
        results.append(1)   # pass
    else:
        results.append(0)   # fail

# putting everything into a dataframe
students_df = pd.DataFrame({
    "StudyHours": hours_studied,
    "Attendance": attendance_percent,
    "QuizScore": quiz_marks,
    "Result": results
})

# --------------------------------------------------
# Preparing data for machine learning
# --------------------------------------------------

features = students_df[["StudyHours", "Attendance", "QuizScore"]]
target = students_df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.25,
    random_state=10
)

# --------------------------------------------------
# Training the model
# --------------------------------------------------

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# checking how well it performs
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

# --------------------------------------------------
# Simple advisory function (acts like an agent)
# --------------------------------------------------

def get_student_advice(hours, attendance, quiz):
    """
    Takes student input and returns prediction + suggestion
    """

    prediction = model.predict([[hours, attendance, quiz]])[0]

    if prediction == 1:
        return "Result: PASS (likely) | Advice: You're doing well, stay consistent."

    # if predicted fail → give basic advice
    if attendance < 75:
        return "Result: At Risk | Advice: Focus on improving attendance first."
    
    if hours < 4:
        return "Result: At Risk | Advice: Increase your daily study time."

    return "Result: At Risk | Advice: Work on both consistency and revision."

# --------------------------------------------------
# Testing with sample students
# --------------------------------------------------

print("\n--- Sample Predictions ---")

print("Student 1:",
      get_student_advice(2, 65, 50))   # low effort

print("Student 2:",
      get_student_advice(8, 90, 85))   # strong student

      # --------------------------------------------------
# Interactive User Input System
# --------------------------------------------------

def run_advisor():
    print("\n--- Student Advisory System ---")
    
    while True:
        try:
            hours = float(input("\nEnter study hours per day: "))
            attendance = float(input("Enter attendance (%): "))
            quiz = float(input("Enter quiz score: "))
            
            result = get_student_advice(hours, attendance, quiz)
            print("\n>>>", result)

        except ValueError:
            print("Please enter valid numeric values.")

        # option to continue
        choice = input("\nDo you want to check another student? (y/n): ").lower()
        if choice != 'y':
            print("Exiting system. Goodbye!")
            break


# run the interactive system
run_advisor()