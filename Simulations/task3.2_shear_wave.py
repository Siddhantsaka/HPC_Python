import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from matplotlib.animation import FuncAnimation

# Lattice parameters
c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
c = np.vstack((c_x, c_y)).T
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

# Function to calculate velocity from distribution function and density
def calculate_velocity(f, density):
    return np.dot(f.T, c).T / density

# Function to calculate total density from distribution function
def calculate_density(f):
    return np.sum(f, axis=0)

# Function to perform streaming step in lattice Boltzmann simulation
def stream(f):
    for i in range(1, 9):
        f[i] = np.roll(f[i], c[i], axis=(0, 1))
    return f

# Function to calculate equilibrium distribution function
def calculate_equilibrium(density, velocity):
    local_velocity_avg = velocity[0, :, :] ** 2 + velocity[1, :, :] ** 2
    cu = np.dot(velocity.T, c.T).T
    velocity_2 = cu ** 2
    f_eq = (((1 + 3 * cu + 9 / 2 * velocity_2 - 3 / 2 * local_velocity_avg) * density).T * W).T
    return f_eq

# Function to perform collision step in lattice Boltzmann simulation
def calculate_collision(f, rl):
    density = calculate_density(f)
    velocity = calculate_velocity(f, density)
    f_eq = calculate_equilibrium(density, velocity)
    f -= rl * (f - f_eq)
    return f, density, velocity

# Function to calculate velocity magnitude for plotting
def v_plot(velocity):
    return np.sqrt(velocity[0, 1:-1, 1:-1]**2 + velocity[1, 1:-1, 1:-1]**2)

# Main function to simulate shear wave decay
def shear_wave_simulation(x_dim=100, y_dim=150, omega=1.0, ep=0.05, save_every=10, steps=10000, relaxation=1):

    # Function to calculate perturbation decay
    def decay_perturbation(t, viscosity):
        size = y_dim
        return ep * np.exp(-viscosity * (2 * np.pi / size) ** 2 * t)

    # Create meshgrid and initial conditions
    x, y = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
    density = 1 + ep * np.sin(2 * np.pi / x_dim * x)
    velocity = np.zeros((2, y_dim, x_dim), dtype=np.float32)
    f = calculate_equilibrium(density, velocity)

    # Lists to store data
    den_list, max_min_list, theoretical_velocity = [], [], []
    d_ani = []
    dens = []

    # Main simulation loop
    for step in range(steps):
        print(f'{step + 1}/{steps}', end="\r")
        f = stream(f)
        f, density, velocity = calculate_collision(f, omega)

        if step % save_every == 0:
            d_ani.append(density)
            plt.clf()
            plt.scatter(x, y, c=density, vmin=np.max(density), vmax=np.max(density))
            plt.xlabel('X meshgrid')
            plt.ylabel('Y meshgrid')
            plt.title('Density flow in shear wave decay')
            plt.colorbar()
            plt.savefig(f'graphs/task3.2/decay_grid/density_decay_{step}.png', pad_inches=1)

        if step % save_every == 0:
            plt.cla()
            plt.ylim([-ep + 1, ep + 1])
            plt.plot(np.arange(x_dim), density[y_dim // 4, :])
            plt.xlabel('X Position')
            plt.ylabel(f'Density ρ (y = {y_dim // 4})')
            plt.grid()
            plt.savefig(f'graphs/task3.2/specific_posn/density_decay_{step}.png', pad_inches=1)

        dens.append(density[y_dim // 4, x_dim // 4] - 1)

        den_list.append(np.max(density - 1))
        max_min_list.append(np.min(density - 1))

    # Analyze simulation results
    den_list = np.array(den_list)
    xl = argrelextrema(den_list, np.greater)[0]
    den_list = den_list[xl]
    xt = np.arange(len(dens))
    plt.figure()
    plt.plot(xt, dens, color='blue', label='Simulated Density')
    plt.xlabel(f'Timestep (ω = {omega})')
    plt.ylabel(f'Density ρ (x = {x_dim // 4}, y = {y_dim // 4})')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'graphs/task3.2/omegadd_{omega}.png', bbox_inches='tight', pad_inches=0.5)
    plt.show()

    # Calculate and compare viscosity
    simulated_viscosity = curve_fit(decay_perturbation, xdata=xl, ydata=den_list)[0][0]
    analytical_viscosity = (1 / 3) * ((1 / omega) - 0.5)

    # Animation of density flow
    def update(i):
        plt.clf()
        plt.title('Flow of density in Shear Wave')
        plt.xlabel('X')
        plt.ylabel('Y')
        im = plt.scatter(x, y, c=d_ani[i], vmin=d_ani[i].min(), vmax=d_ani[i].max())
        return im

    plt.clf()
    plt.scatter(x, y, c=density, vmin=density.min(), vmax=density.max())
    plt.xlabel('X meshgrid')
    plt.ylabel('Y meshgrid')
    plt.title('Density flow in shear wave decay')
    plt.colorbar()
    animation = FuncAnimation(plt.gcf(), update, frames=len(d_ani), interval=200, blit=False, repeat=False)
    animation.save("graphs/task3.2/density flow in shear wave decay.gif", writer="pillow")
    plt.show()

    return simulated_viscosity, analytical_viscosity

# Call the simulation function
shear_wave_simulation()
