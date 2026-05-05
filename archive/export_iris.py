import pandas as pd
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()

# Create a DataFrame with the features
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Custom Subculture Names
subcultures = {0: "Eshay", 1: "Lad", 2: "Tradie"}

# Add the target labels (species names instead of just numbers 0,1,2 for readability)
df['species_id'] = iris.target
df['subculture_name'] = [subcultures[i] for i in iris.target]

# Save to CSV
csv_filename = "iris_dataset.csv"
df.to_csv(csv_filename, index=False)

print(f"Successfully saved {csv_filename}!")
print("First 5 rows:")
print(df.head())
