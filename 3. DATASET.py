import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_classes = 10
samples_per_class = 100
min_distance = 4  # minimum distance between cluster centers (no overlap)
cluster_centers = []

# Step 1: Generate 10 well-separated but close cluster centers
while len(cluster_centers) < num_classes:
    candidate = np.random.uniform(-10, 10, size=2)
    if all(np.linalg.norm(candidate - c) >= min_distance for c in cluster_centers):
        cluster_centers.append(candidate)

X = []
y = []

# Step 2: Create curved data around each cluster center
for digit, center in enumerate(cluster_centers):
    angle = np.linspace(0, 2 * np.pi, samples_per_class)
    radius = 1.2 + 0.1 * digit  # increasing slightly to avoid overlap

    x1 = center[0] + radius * np.cos(angle) + np.random.normal(0, 0.15, samples_per_class)
    x2 = center[1] + radius * np.sin(angle**2) + np.random.normal(0, 0.15, samples_per_class)

    X.append(np.stack([x1, x2], axis=1))
    y.append(np.full(samples_per_class, digit))

# Step 3: Stack and save the dataset
X = np.vstack(X)
y = np.hstack(y)
dataset = np.hstack([X, y.reshape(-1, 1)])

# Save as .txt file
np.savetxt("dataset.txt", dataset, delimiter=",", fmt="%.4f")

# Step 4: Plot the result
plt.figure(figsize=(12, 9))
for digit in range(num_classes):
    plt.scatter(X[y == digit][:, 0], X[y == digit][:, 1], label=f"Digit {digit}", s=15)
plt.title("Digit Dataset: Curved, Non-overlapping, Close Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
