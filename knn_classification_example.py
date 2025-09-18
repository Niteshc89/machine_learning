from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target # Target variable (species)

# 2. Split the data into training and testing sets
# test_size=0.3 means 30% of the data will be used for testing
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create a KNN classifier instance
# n_neighbors specifies the 'k' in KNN (number of neighbors to consider)
knn = KNeighborsClassifier(n_neighbors=5)

# 4. Train the model
# The model learns from the training data
knn.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = knn.predict(X_test)

# 6. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the KNN model: {accuracy:.2f}")

# 7. Predict a new, unseen data point
# Example: a new flower with sepal length, sepal width, petal length, petal width
new_flower = [[5.1, 3.5, 1.4, 0.2]]
predicted_species_index = knn.predict(new_flower)
predicted_species_name = iris.target_names[predicted_species_index[0]]
print(f"Predicted species for {new_flower}: {predicted_species_name}")