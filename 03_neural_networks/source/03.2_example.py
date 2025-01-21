from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# example dataset: 2D points with two classes (a and b) seperated by a circle
np.random.seed(42)  # For reproducibility

class_a=[]
class_b=[]

n=500

x1=5.
y1=5.

class Seperator:
    def __init__(self,radius):
        self.radius=radius
    def __call__(self,x,y):
        if pow(x,2)+pow(y,2)>pow(self.radius,2):
            return True
        return False

seperator=Seperator(2.5)
    
while len(class_a)<n:
    x=np.random.uniform(-x1, x1)
    y=np.random.uniform(-y1, y1)
    if seperator(x,y):
        class_a.append([x,y])

while len(class_b)<n:
    x=np.random.uniform(-x1, x1)
    y=np.random.uniform(-y1, y1)
    if not seperator(x,y):
        class_b.append([x,y])

class_a=np.array(class_a)
class_b=np.array(class_b)
        
# Combine data
X = np.vstack((class_a, class_b))  # Features (x1, x2)
y = np.hstack((np.zeros(n), np.ones(n)))  # Labels (0 for class 'a', 1 for class 'b')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network
clf = MLPClassifier(hidden_layer_sizes=(2,),  # One hidden layer with 2 nodes
                      activation='relu',       # ReLU activation function
                     solver='adam',           # Adam optimizer
                     max_iter=1000,           # Maximum iterations
                     random_state=42)

# Train the neural network
clf.fit(X_train, y_train)

# Predict on the test set
#y_pred = clf.predict(X_test)
#y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities for class 'b'

# Evaluate the model
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy:.2f}")

# Print probabilities for some test points
#for i, prob in enumerate(y_prob[:5]):
#    print(f"Point {X_test[i]} -> Probability of class 'b': {prob:.2f}")

# Generate heatmap of predictions
# Define the grid for the heatmap
x_range = np.linspace(-5, 5, 100)  # 100 points in the x-direction
y_range = np.linspace(-5, 5, 100)  # 100 points in the y-direction
xx, yy = np.meshgrid(x_range, y_range)  # Create a grid of x, y coordinates

# Combine grid points into a feature matrix for prediction
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Predict probabilities for each grid point
z = clf.predict_proba(grid_points)[:, 1]  # Probability of class 'b'
z = z.reshape(xx.shape)  # Reshape to match the grid dimensions

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, z, levels=50, cmap='coolwarm', alpha=0.8)  # Heatmap
plt.colorbar(label='Probability of class b')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Heatmap of Predictions')
plt.scatter(class_a[:, 0], class_a[:, 1], c='blue', label='Class a', edgecolor='k')
plt.scatter(class_b[:, 0], class_b[:, 1], c='red', label='Class b', edgecolor='k')
plt.legend()
plt.show()
