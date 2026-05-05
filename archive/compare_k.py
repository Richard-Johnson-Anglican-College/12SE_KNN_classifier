from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
X, y = load_iris(return_X_y=True)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- TESTING DIFFERENT K VALUES ---")
print("We are changing 'n_neighbors' to see how it affects accuracy.\n")

# Try different values of K
for k in [1, 5, 20, 50, 100]:
    # Create model with specific K
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train
    knn.fit(X_train, y_train)
    
    # Check Accuracy
    acc = accuracy_score(y_test, knn.predict(X_test))
    
    print(f"When K = {k:3} | Accuracy = {acc*100:6.2f}%")

print("\nNotice: When K gets too large (like 100), the model starts to")
print("lose accuracy because it's looking at too many 'neighbors'!")
