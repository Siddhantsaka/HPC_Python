import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.interpolate import make_interp_spline
from matplotlib import animation

# Temp Constants and parameters
x_dim = 100
y_dim = 150
c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
c = np.vstack((c_x, c_y)).T
density = np.zeros((y_dim, x_dim), dtype=np.float32)
velocity = np.zeros((2, x_dim, y_dim), dtype=np.float32)
f = np.random.uniform(2, size=(9, y_dim, x_dim))
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

omega = 1.0
ep = 0.05

# Calculate velocity
def calculate_velocity(f, density):
    velocity = (np.dot(f.T, c).T / density)
    return velocity


# Calculate density
def calculate_density(f):
    return np.sum(f, axis=0)


# Streaming step in Lattice Boltzmann method
def stream(f):
    for i in range(1,9):
        f[i] = np.roll(f[i],c[i], axis=(0, 1))
    return f


# Calculate equilibrium distribution function
def calculate_equilibrium(density, velocity):
    local_velocity_avg = velocity[0, :, :] ** 2 + velocity[1, :, :] ** 2
    cu = np.dot(velocity.T, c.T).T
    #print(cu.T.shape)
    velocity_2 = cu ** 2
    #print(velocity_2.T.shape)
    f_eq = (((1 + 3 * cu + 9 / 2 * velocity_2 - 3 / 2 * local_velocity_avg) * density).T * W).T
    return f_eq


# Perform collision step
def calculate_collision(f, rl):
    density = calculate_density(f)
    velocity = calculate_velocity(f, density)
    f_eq = calculate_equilibrium(density, velocity)
    f -= rl * (f - f_eq)
    return f, density, velocity

# Calculate magnitude of velocity for plotting
def v_plot(velocity):
    v = np.sqrt(velocity[0, 1:-1, 1:-1]**2 + velocity[1, 1:-1, 1:-1]**2)
    return v


# Shear wave simulation function
def shear_wave_simulation(x_dim=100, y_dim=100, omega = 0.5, ep = 0.05, save_every = 20, steps = 1000):

    # Define functions for theoretical and perturbation calculations
    def instant_theoretical_velocity(v):
        y = np.exp(-v * (2 * np.pi / y_dim) ** 2)
        return y


    def decay_perturbation(t, viscosity):
        size = y_dim
        return ep * np.exp(-viscosity * (2 * np.pi / size) ** 2 * t)

    # Initialize variables
    density = np.ones((y_dim, x_dim), dtype=np.float32)
    velocity = np.zeros((2, y_dim, x_dim), dtype=np.float32)
    velocity[1, :, :] = ep * np.sin(2 * np.pi / y_dim * np.arange(y_dim)[:, np.newaxis])
    f = calculate_equilibrium(density, velocity)
    print(f)

    # Lists to store data
    vel_list, max_min_list,  theoretical_velocity = [], [], []
    vn= velocity
    velocity_decay_data = []

    # Main simulation loop
    for step in range(steps):
        print(f'{step + 1}//{steps}', end="\r")
        f = stream(f)
        f, density, velocity = calculate_collision(f, omega)

        # Plot and save velocity magnitude
        if step % save_every == 0:
            plt.clf()
            X, Y = np.meshgrid(np.arange(y_dim-2), np.arange(x_dim-2))
            v_mag = v_plot(velocity)
            plt.scatter(X, Y, c=v_mag.T, vmin=np.min(v_mag.T), vmax=np.max(v_mag.T))
            plt.title("Shear wave step".format(step))
            plt.xlabel("x-Position")
            plt.ylabel("y-Position")
            fig = plt.colorbar()
            fig.set_label("Velocity", rotation=270, labelpad=15)
            plt.savefig(f'shear wave step {step}')

        kinematic_viscosity = 1 / 3 * (1 / omega - 1 / 2)
        y_val = instant_theoretical_velocity(kinematic_viscosity)
        y_val = y_val * (vn[1, x_dim // 4, :])
        vn[1, x_dim // 4, :] = y_val
        theoretical_velocity.append(y_val.max())

        vel_list.append(np.max(velocity[1, :, :]))
        max_min_list.append(np.min(velocity[1, :, :]))

        # Plot and save velocity decay lines
        if step % save_every == 0:
            plt.ylim([-ep, ep])
            plt.plot(np.arange(y_dim), velocity[1, :, x_dim // 4])
            plt.xlabel('Y Position')
            plt.ylabel(f'Velocity u(x = {x_dim // 4},y)')
            plt.grid()
            velocity_decay_data.append(velocity[1, :, x_dim // 4])  # Store the velocity data

    # Plot and save combined velocity decay graph
    plt.clf()
    plt.ylim([-ep, ep])
    for data in velocity_decay_data:
        plt.plot(np.arange(y_dim), data, alpha=0.5)

    plt.xlabel('Y Position')
    plt.ylabel(f'Velocity u(x = {x_dim // 4},y)')
    plt.grid()
    plt.savefig(f'./velocity_decay_combined.png', pad_inches=1)
    plt.close()

    # Plot and save simulated vs analytical viscosity comparison
    x = np.arange(steps)
    plt.xlim([0, len(x)])
    X_Y_Sp = make_interp_spline(np.arange(steps), vel_list)
    X_Y_Sp_1 = make_interp_spline(np.arange(steps), max_min_list)
    n=np.arange(steps)
    X_ = np.linspace(0, n[-1], 500)
    Y_ = X_Y_Sp(X_)
    Y_1 = X_Y_Sp_1(X_)
    plt.plot(X_, Y_, color='blue')
    plt.plot(X_, Y_1, color='blue')
    plt.plot(x, theoretical_velocity, color='black', linestyle='dotted', linewidth=3)
    plt.fill_between(X_, Y_, Y_1, color="blue", alpha=.2)
    plt.xlabel('Time evolution')
    plt.ylabel(f'Velocity at (y = {y_dim // 4})')
    plt.legend(['Simulated Maxima', 'Simulated Minima', 'Analytical ux(y=25)'])
    plt.grid()
    plt.savefig(f'./omega_{omega}.png', pad_inches=1)
    plt.close()

    # Perform curve fitting to estimate simulated viscosity
    simulated_viscosity = curve_fit(decay_perturbation, xdata=x, ydata=vel_list)[0][0]
    analytical_viscosity = (1 / 3) * ((1 / omega) - 0.5)

    return simulated_viscosity, analytical_viscosity

simulated_viscosity, analytical_viscosity = shear_wave_simulation()
print("Simulated Viscosity:", simulated_viscosity)
print("Analytical Viscosity:", analytical_viscosity)

ov=np.arange(0.1,2,0.2)
sim_ves=[]
ana_ves=[]

for i in ov:
    a,b=shear_wave_simulation(x_dim=100, y_dim=100, omega = i, ep = 0.05, save_every = 20, steps = 1000)
    sim_ves.append(a)
    ana_ves.append(b)


plt.plot(ov,sim_ves,color='red', label='Simulated')
plt.plot(ov,ana_ves,color='black',linestyle = 'dotted', linewidth= 2,label='Calculated')

plt.title("Kinematic viscosity vs Omega")
plt.xlabel("Omega")
plt.ylabel("Kinematic Viscosity")
plt.legend()
plt.grid()
plt.savefig('Kinematic viscosity vs Omega 1000')
plt.show()


