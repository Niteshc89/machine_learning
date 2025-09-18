import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Generate sample data
# Let's create a simple dataset where y is a function of x with some noise
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create and train the KNN Regressor model
# We choose n_neighbors=5, meaning we consider the 5 closest neighbors
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = knn_regressor.predict(X_test)

# 5. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Optional: Visualize the predictions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data (actual)')
plt.scatter(X_test, y_pred, color='red', marker='x', s=100, label='Test data (predicted)')
plt.title('KNN Regression Example')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()