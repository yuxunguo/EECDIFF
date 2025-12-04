import matplotlib.pyplot as plt
import numpy as np

# Define poles
poles = [0, 2]

# Contour: vertical line N = c + i t
c = 1.0
t = np.linspace(-5, 5, 400)
N_contour = c + 1j * t

# Plot
plt.figure(figsize=(6,6))

# Plot contour
plt.plot(N_contour.real, N_contour.imag, 'b', label='Mellin contour')

# Plot poles
for p in poles:
    plt.plot(p, 0, 'ro', markersize=10, label=f'Pole at N={p}')

# Add arrows to indicate direction
plt.arrow(c, -4.5, 0, 0.5, head_width=0.1, head_length=0.2, color='blue')
plt.arrow(c, 4.0, 0, 0.5, head_width=0.1, head_length=0.2, color='blue')

# Labels and grid
plt.xlabel('Re(N)')
plt.ylabel('Im(N)')
plt.title('Inverse Mellin Transform Contour')
plt.axhline(0, color='k', linestyle='--')
plt.grid(True)
plt.xlim(-1, 3)
plt.ylim(-5.5, 5.5)
plt.legend()
plt.show()
