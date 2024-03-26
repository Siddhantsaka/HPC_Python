import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to perform streaming step in lattice Boltzmann method, updating the distribution function 'f'
def stream(f):
    # Loop through each direction of the lattice velocities
    for i in range(1, 9):
        # Roll the distribution function in the direction of the velocity vectors
        f[i] = np.roll(f[i], c[i], axis=(0, 1))
    return f


# Function to Calculate velocity from distribution function and density
def calculate_velocity(f, density):
    # Velocity is calculated as the weighted sum of the distribution functions along each direction, divided by the
    # density
    velocity = (np.dot(f.T, c).T / density)
    return velocity

# Calculate magnitude of velocity for visualization
def v_plot(velocity):
    # Calculate the magnitude of the 2D velocity vector at each point in the field
    v = np.sqrt(velocity[0, :, :] ** 2 + velocity[1, :, :] ** 2)
    return v


# Calculate density of distribution
def calculate_density(f):
    return np.sum(f, axis=0)


# Set dimensions and velocity vectors for the simulation
x_dim = 10
y_dim = 15
c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
c = np.vstack((c_x, c_y)).T
density = np.zeros((x_dim, y_dim), dtype=np.float32)
velocity = np.zeros((2, x_dim, y_dim), dtype=np.float32)
f = np.random.normal(2, size=(9, x_dim, y_dim))

# Calculate initial density and velocity
density = calculate_density(f)
velocity = calculate_velocity(f, density)


# Function to create a scatter plot of density values
def density_graph():
    global density
    # Extract non-zero density positions and their corresponding values
    positions = np.argwhere(density)
    density_values = density[positions[:, 0], positions[:, 1]]
    circle_sizes = density_values * 5  # Scale circle sizes for visualization

    # Create and display the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(positions[:, 1], positions[:, 0], s=circle_sizes, c=density_values, cmap='viridis', alpha=1)
    plt.colorbar(label='Density Value')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.title('Scatter Plot of Density Values with Circles')
    plt.show()

    # Get the positions (indices) and density values from the matrix
    positions = np.argwhere(density)
    density_values = density[positions[:, 0], positions[:, 1]]

    # Calculate initial circle sizes based on density values
    circle_sizes = density_values * 20  # Adjust the scaling factor as needed

    # Create the scatter plot with circles (initial frame)
    fig, ax = plt.subplots()
    sc = ax.scatter(positions[:, 1], positions[:, 0], s=circle_sizes, c=density_values, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Density Value')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.title('Animated Scatter Plot of Density Values')
    plt.tight_layout()

    # Function to update the scatter plot for each animation frame
    def update(frame):
        # Modify the density values for the next frame (replace with your data modification logic)
        global f, density, velocity
        f = stream(f)
        density = calculate_density(f)
        velocity = calculate_velocity(f, density)
        modified_density = density + np.random.normal(scale=0.05, size=(10, 15))

        # Update the density values and circle sizes
        density_values = modified_density[positions[:, 0], positions[:, 1]]
        circle_sizes = density_values * 10

        # Update the scatter plot data
        sc.set_sizes(circle_sizes)
        sc.set_array(density_values)

        return sc,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=100, interval=200, blit=True)

    plt.show()


def velocity_graph():
    global f, density, velocity
    # Generate a grid for plotting
    X, Y = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
    fig, ax = plt.subplots()

    # Create an initial stream plot for the velocity field
    im = ax.streamplot(X, Y, velocity[0, :, :].T, velocity[1, :, :].T, color=v_plot(velocity).T)
    fig.colorbar(im.lines).set_label("Velocity", rotation=270, labelpad=15)

    # Animation update function
    def update_plot(i):
        global f, density, velocity

        # Update distribution function, density, and velocity
        f = stream(f)
        density = calculate_density(f)
        velocity = calculate_velocity(f, density)

        # Clear the plot and update with new data
        ax.clear()
        im = ax.streamplot(X, Y, velocity[0, :, :].T, velocity[1, :, :].T, color=v_plot(velocity).T)
        ax.set_title(f"Frame {i + 1}")
        plt.xlabel("x-Position")
        plt.ylabel("y-Position")

        return im

    # Create the animation
    animation = FuncAnimation(fig, update_plot, frames=17, interval=20, blit=False)
    plt.show()

# Prompt user to choose the type of plot to display: density or velocity
usr_ipt = input('Enter the type of plot: density_graph(1) or Velocity_graph(2)  ')

if usr_ipt == '1':
    density_graph()
else:
    velocity_graph()
