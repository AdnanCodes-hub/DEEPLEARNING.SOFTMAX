import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(r"C:\Users\HMC\Desktop\FINAL\dataset.txt", delimiter=",")
X = data[:, :2]
Y = data[:, 2]

# Dataset Visualization
plt.figure(figsize=(7, 5))
scatter = plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', edgecolors='k')
plt.title("Dataset Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
plt.gca().add_artist(legend1)
plt.grid(True)
plt.tight_layout()
plt.show()

# Build and Train the Model
model = Sequential([
    Dense(units=10, activation='relu', input_shape=(2,)),
    Dense(units=10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(X, Y, epochs=100)

# Save the model
model_path = r"C:\Users\HMC\Desktop\FINAL\softmax_model.keras"
model.save(model_path)

# Plot decision boundaries
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid)
    Z = np.argmax(preds, axis=1).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Softmax Decision Boundary")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_decision_boundary(X, Y, model)
