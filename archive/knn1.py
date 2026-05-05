print("Loading machine learning libraries (this may take a few seconds)...")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
print("Libraries loaded successfully!\n")

# 1. Load the data
print("Loading Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split into Training (80%) and Testing (20%)
# We keep the test set secret so we can check if the model actually learned
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. SCALE THE DATA (The most important step for KNN!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Create the Model (Let's look at the 5 nearest neighbors)
knn = KNeighborsClassifier(n_neighbors=5)

# 5. Train it
print("Training the KNN model...")
knn.fit(X_train, y_train)

# 6. Predict and Check Accuracy
print("Making predictions on test data...")
predictions = knn.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")