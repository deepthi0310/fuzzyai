import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib

df = pd.read_csv("dosage_dataset.csv")

X = df[['ph', 'turbidity', 'temperature']]  # features
y = df['dosage']  # label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

joblib.dump(model, 'ml_model.pkl')
