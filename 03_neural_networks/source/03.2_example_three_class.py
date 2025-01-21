from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Generate the dataset
np.random.seed(42)
num_points = 200

# Generate points for each class
radii_a = np.random.rand(num_points) * 2 * np.pi
distances_a = np.random.rand(num_points) 

radii_b = np.random.rand(num_points) * np.pi
distances_b = np.random.rand(num_points) + 1.2

radii_c = np.pi+np.random.rand(num_points) * np.pi
distances_c = np.random.rand(num_points) + 1.2

coords_0 = np.column_stack((distances_a * np.cos(radii_a),distances_a * np.sin(radii_a)))
coords_1 = np.column_stack((distances_b * np.cos(radii_b),distances_b * np.sin(radii_b)))
coords_2 = np.column_stack((distances_c * np.cos(radii_c),distances_c * np.sin(radii_c)))




# Combine all classes
X = np.vstack((coords_0, coords_1, coords_2))
y = np.hstack((np.zeros(len(coords_0)), np.ones(len(coords_1)), np.full(len(coords_2), 2)))


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network
clf = MLPClassifier(hidden_layer_sizes=(16, 8, 4), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Train the neural network
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")





# Generate heatmaps
x_range = np.linspace(-2, 2, 200)
y_range = np.linspace(-2, 2, 200)
xx, yy = np.meshgrid(x_range, y_range)
grid_points = np.c_[xx.ravel(), yy.ravel()]
probabilities = clf.predict_proba(grid_points)

# Generate heatmaps for each class
width_cm = 3.5
dpi = 300
width_inch = width_cm * 0.3937
height_inch = width_inch * (6 / 8)

for i, class_name in enumerate(["Class 0", "Class 1", "Class 2"]):
    plt.figure(figsize=(width_inch, height_inch))
    z = probabilities[:, i].reshape(xx.shape)
    heatmap=plt.contourf(xx, yy, z, levels=50, cmap='coolwarm', alpha=0.8, vmin=0, vmax=1)
    cbar = plt.colorbar(heatmap, ticks=[0.0, 0.25, 0.5, 0.75,1.0])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.savefig(f"heatmap_{class_name.lower().replace(' ', '_')}.png", dpi=dpi, bbox_inches='tight')
    plt.close()

print("Heatmaps saved for each class.")
