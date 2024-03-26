import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_dim = 10
y_dim = 15
c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
c = np.vstack((c_x, c_y)).T
density = np.zeros((y_dim, x_dim), dtype=np.float32)
velocity = np.zeros((2, x_dim, y_dim), dtype=np.float32)
f = np.random.uniform(2, size=(9, y_dim, x_dim))
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
relaxation = 0.7

def stream(f):
    for i in range(1,9):
        f[i] = np.roll(f[i],c[i], axis=(0, 1))
        density = np.sum(f, axis=0)
        velocity = (np.dot(f.T, c).T / density)
    return calculate_collision(f, density, velocity, relaxation)



# Calculate velocity from distribution function and density
def calculate_velocity(f, density):
    velocity = (np.dot(f.T, c).T / density)
    return velocity

# Calculate magnitude of velocity for visualization
def v_plot(velocity):
    v = np.sqrt(velocity[0, :, :]**2 + velocity[1, :, :]**2)
    #print(np.shape(v))
    return v


# Calculate density of distribution
def calculate_density(f):
    return np.sum(f, axis=0)


def calculate_collision(f, density, velocity, rl):
    f_eq = calculate_equilibrium(density, velocity)
    f -= rl * (f-f_eq)
    return f, density, velocity


def calculate_equilibrium(density, velocity):
    local_velocity_avg = velocity[0, :, :] ** 2 + velocity[1, :, :] ** 2
    cu = np.dot(velocity.T, c.T).T
    velocity_2 = cu ** 2
    f_eq = (((1 + 3 * cu + 9 / 2 * velocity_2 - 3 / 2 * local_velocity_avg) * density).T * W).T
    return f_eq

f,density,velocity=stream(f)



# Function to create a scatter plot of density values
def density_graph():
    global density
    positions = np.argwhere(density)
    density_values = density[positions[:, 0], positions[:, 1]]
    circle_sizes = density_values * 5

    # Create the scatter plot with circles
    plt.figure()
    plt.scatter(positions[:, 1], positions[:, 0], s=circle_sizes, c=density_values, cmap='viridis', alpha=1)
    plt.colorbar(label='Density Value')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.title('Scatter Plot of Density Values with Circles')
    plt.tight_layout()
    plt.show()


    # Get the positions (indices) and density values from the matrix
    positions = np.argwhere(density)
    density_values = density[positions[:, 0], positions[:, 1]]

    # Calculate initial circle sizes based on density values
    circle_sizes = density_values * 10  # Adjust the scaling factor as needed

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
        f,den,velocity = stream(f)
        plt.cla()
        modified_density = den
        # Update the density values and circle sizes
        density_values = modified_density[positions[:, 0], positions[:, 1]]
        circle_sizes = density_values * 10

        # Update the scatter plot data
        sc = ax.scatter(positions[:, 1], positions[:, 0], s=circle_sizes, c=density_values, cmap='viridis', alpha=0.7)
        return sc,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=100, interval=200, blit=True)

    plt.show()


def velocity_graph():
    global f, density, velocity
    # Set up the grid for the streamplot
    X, Y = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
    print(np.shape(X))
    print(np.shape(Y))
    print(np.shape(v_plot(velocity)))
    fig, ax = plt.subplots()

    # Create the initial streamplot
    im = ax.streamplot(X, Y, velocity[0, :, :], velocity[1, :, :], color=v_plot(velocity))
    fig.colorbar(im.lines).set_label("Velocity", rotation=270, labelpad=15)

    # Animation update function
    def update_plot(i):
        global f, density, velocity

        # Update distribution function, density, and velocity
        f, density, velocity = stream(f)

        # Clear the plot and update with new data
        ax.clear()
        im = ax.streamplot(X, Y, velocity[0, :, :], velocity[1, :, :], color=v_plot(velocity))
        ax.set_title(f"Frame {i + 1}")
        plt.xlabel("x-Position")
        plt.ylabel("y-Position")

        return im

    # Create the animation
    animation = FuncAnimation(fig, update_plot, frames=17, interval=20, blit=False)
    plt.show()


# Get user input to choose the type of plot
usr_ipt = input('Enter the type of plot: density_graph(1) or Velocity_graph(2)  ')

if usr_ipt == '1': density_graph()
else: velocity_graph()
