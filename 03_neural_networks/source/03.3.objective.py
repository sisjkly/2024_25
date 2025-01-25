import numpy as np
import matplotlib.pyplot as plt

# Define the function with adjusted quadratic term coefficient
def objective_function(x):
    return 0.2 * (x**4) - 0.3 * (x + 1.4)**3 - 0.5 * (x**2) + 0.6 * x + 10

# Generate x values with the updated range
x = np.linspace(-2.5, 4.25, 500)

# Compute y values
y = objective_function(x)

width_inch = 3
height_inch = 2  # Adjust this value as needed to maintain aspect ratio

# Create the plot
plt.figure(figsize=(width_inch, height_inch))

plt.plot(x, y, color="blue", linewidth=2)
plt.xlabel(r"$\theta$", fontsize=14)  # Label for x-axis
plt.ylabel(r"$\mathcal{L}$", fontsize=14)  # Label for y-axis
plt.xticks([])  # Remove x-axis tick marks
plt.yticks([])  # Remove y-axis tick marks
plt.grid(alpha=0.3)  # Add a light grid for better visibility
plt.ylim([-2.5, 12.5])

plt.savefig('03.3_objective.png', dpi=300, bbox_inches='tight')
plt.show()

plt.show()
