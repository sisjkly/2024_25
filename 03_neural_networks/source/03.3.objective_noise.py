import numpy as np
import matplotlib.pyplot as plt

# Define the original function
def objective_function(x, coeffs):
    c, d= coeffs
    return 0.2 * (x**4) - 0.3 * (x + 1.4)**3 + c * (x**2) + d * x + 10

# Original coefficients
original_coeffs = [ -0.5, 0.6]

# Generate x values
x = np.linspace(-2.5, 4.25, 500)

# Plot the original function
y_original = objective_function(x, original_coeffs)


width_inch = 3
height_inch = 2 

plt.figure(figsize=(width_inch, height_inch))

plt.plot(x, y_original, color="blue", linewidth=2)

# Set the standard deviation for noise
noise_std = 0.2

# Plot noisy variations
for _ in range(10):
    noisy_coeffs = original_coeffs + np.random.normal(0, noise_std, size=len(original_coeffs))
    y_noisy = objective_function(x, noisy_coeffs)
    plt.plot(x, y_noisy, color="red", linewidth=1, alpha=0.7)

# Customize the plot
plt.xlabel(r"$\theta$", fontsize=14)
plt.ylabel(r"$\mathcal{L}(B)$", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.grid(alpha=0.3)
plt.legend(frameon=False)
plt.ylim([-2.5, 12.5])
plt.savefig('03.3_objective_noise.png', dpi=300, bbox_inches='tight')
plt.show()
