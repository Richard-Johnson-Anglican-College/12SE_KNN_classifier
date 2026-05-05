import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 1. Load data
iris = load_iris()
X = iris.data[:, :2] # Use first two measurements for the 2D map
y = iris.target

# 2. Pick a "mystery" flower
mystery_point = np.array([[6.1, 2.7]])

# 3. LET THE BRAIN GUESS (KNN Algorithm)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y) # Memorize the map
prediction_index = knn.predict(mystery_point)[0]
subcultures = {0: "Eshay", 1: "Lad", 2: "Tradie"}
prediction_name = subcultures[prediction_index]

# 4. Create the Visual Map
plt.figure(figsize=(10, 7))

# Plot the known flowers
colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
for i, color in enumerate(colors):
    plt.scatter(X[y == i, 0], X[y == i, 1], color=color, 
                label=f'Subculture: {subcultures[i]}', 
                edgecolors='k', s=100, alpha=0.7)

# Plot the mystery star
plt.scatter(mystery_point[:, 0], mystery_point[:, 1], color='red', marker='*', 
            s=400, edgecolors='black', label='Mystery Person')

# Add the prediction label on the graph
plt.text(mystery_point[0, 0] + 0.1, mystery_point[0, 1] + 0.1, 
         f"Prediction: {prediction_name}!", 
         fontsize=14, fontweight='bold', color='red', 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='red'))

# Draw the neighborhood circle
circle = plt.Circle((mystery_point[0, 0], mystery_point[0, 1]), 0.35, 
                    color='red', fill=False, linestyle='--', linewidth=2)
plt.gca().add_patch(circle)

plt.xlabel('Measurement 1 (Sepal Length)')
plt.ylabel('Measurement 2 (Sepal Width)')
plt.title('K-Nearest Neighbors (KNN): Final Prediction', fontsize=16)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.4)

# Save the result
plt.savefig('knn_visual.png', bbox_inches='tight', dpi=150)
print(f"Prediction Complete! The Mystery Flower is: {prediction_name}")
