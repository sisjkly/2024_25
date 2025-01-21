from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Example dataset: Class 'a' inside a circle, class 'b' outside
np.random.seed(42)  # For reproducibility
num_points = 250

# Generate points
radii_a = np.random.rand(num_points) * 2 * np.pi
radii_b = np.random.rand(num_points) * 2 * np.pi
distances_a = np.random.rand(num_points) * 1.0  # Inside the circle
distances_b = 1.2 + np.random.rand(num_points) * 1.0  # Outside the circle

# Class 'a' (inside)
class_a = np.column_stack((distances_a * np.cos(radii_a), distances_a * np.sin(radii_a)))

# Class 'b' (outside)
class_b = np.column_stack((distances_b * np.cos(radii_b), distances_b * np.sin(radii_b)))

# Combine data
X = np.vstack((class_a, class_b))  # Features (x1, x2)
y = np.hstack((np.zeros(len(class_a)), np.ones(len(class_b))))  # Labels (0 for class 'a', 1 for class 'b')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network with three hidden layers
#clf = MLPClassifier(hidden_layer_sizes=(8, 4 ,2),
clf = MLPClassifier(hidden_layer_sizes=(4, 2),
                     activation='relu',           # ReLU activation function
                     solver='adam',               # Adam optimizer
                     max_iter=1000,               # Maximum iterations
                     random_state=42)

# Train the neural network
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities for class 'b'

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate heatmap of predictions
# Define the grid for the heatmap
x_range = np.linspace(-2, 2, 200)  # 200 points in the x-direction
y_range = np.linspace(-2, 2, 200)  # 200 points in the y-direction
xx, yy = np.meshgrid(x_range, y_range)  # Create a grid of x, y coordinates

# Combine grid points into a feature matrix for prediction
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict probabilities for each grid point
z = 1-clf.predict_proba(grid_points)[:, 1]  # Probability of class 'a'
z = z.reshape(xx.shape)  # Reshape to match the grid dimensions

width_cm = 6
dpi = 300  # Resolution in dots per inch

# Convert width to inches
width_inch = width_cm * 0.3937
height_inch = width_inch * (6 / 8)  # Maintain the aspect ratio of (8, 6)

# Create and save the plot
plt.figure(figsize=(width_inch, height_inch))
heatmap=plt.contourf(xx, yy, z, levels=50, cmap='coolwarm', alpha=0.8 , vmin=0., vmax=1.)  # Heatmap
cbar = plt.colorbar(heatmap, ticks=[0.0, 0.2, 0.4, 0.6, 0.8,1.0])

#plt.colorbar(label='Probability of class b')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
#plt.scatter(class_a[:, 0], class_a[:, 1], c='blue', label='Class a', edgecolor='k')
#plt.scatter(class_b[:, 0], class_b[:, 1], c='red', label='Class b', edgecolor='k')

plt.savefig("two_layer_heatmap.png", dpi=dpi, bbox_inches='tight')
plt.legend(frameon=False)
plt.show()
