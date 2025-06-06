import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


file_path = r"C:\codsoft\advertising.csv"
df = pd.read_csv(file_path)

print("Dataset Preview:")
print(df.head())

X = df[['TV', 'Radio', 'Newspaper']]  
y = df['Sales']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nPredicted Sales:")
print(y_pred)
print("\nMean Squared Error:", mse)
print("R-squared:", r2)
