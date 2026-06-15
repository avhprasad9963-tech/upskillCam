import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 95, 100],
    'Marks': [40, 45, 50, 55, 60, 70, 80, 90, 95]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

study_hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance Percentage: "))

prediction = model.predict([[study_hours, attendance]])

print("Predicted Marks:", round(prediction[0], 2))
