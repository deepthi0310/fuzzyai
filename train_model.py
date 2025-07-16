import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Fake training data
data = {
    'ph': [6, 7, 8, 9],
    'turbidity': [100, 80, 60, 40],
    'temperature': [25, 26, 27, 28],
    'dosage': [10, 9, 7, 5]
}
df = pd.DataFrame(data)

X = df[['ph', 'turbidity', 'temperature']]
y = df['dosage']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'model.pkl')  # This saves the file
print("✅ Model trained and saved as model.pkl")
