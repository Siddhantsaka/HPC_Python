import numpy as np
import matplotlib.pyplot as plt

x_dim = 100
y_dim = 50
c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
c = np.vstack((c_x, c_y)).T
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]).T
relaxation = 0.5

den_null = 1
diff = 0.001
shear_viscosity = (1 / relaxation - 0.5) / 3


def calculate_velocity(f, density):
    velocity = (np.dot(f.T, c).T / density)
    return velocity


def calculate_density(f):
    return np.sum(f, axis=0)


def calculate_collision(f, density, velocity, relaxation):
    f_eq = calculate_equilibrium(density, velocity)
    f -= relaxation * (f - f_eq)
    return f


def stream(f):
    for i in range(1, 9):
        f[i] = np.roll(f[i], c[i], axis=(0, 1))
    return f


def bounce_back(grid, uw):
    grid[2, 1:-1, 1] = grid[4, 1:-1, 0]
    grid[5, 1:-1, 1] = grid[7, 1:-1, 0]
    grid[6, 1:-1, 1] = grid[8, 1:-1, 0]
    # for top y = -1
    grid[4, 1:-1, -2] = grid[2, 1:-1, -1]
    grid[7, 1:-1, -2] = grid[5, 1:-1, -1] - 1 / 6 * uw
    grid[8, 1:-1, -2] = grid[6, 1:-1, -1] + 1 / 6 * uw
    return grid


def calculate_equilibrium(density, velocity):
    local_velocity_avg = velocity[0, :, :] ** 2 + velocity[1, :, :] ** 2
    cu = np.dot(velocity.T, c.T).T
    velocity_2 = cu ** 2
    f_eq = (((1 + 3 * cu + 9 / 2 * velocity_2 - 3 / 2 * local_velocity_avg) * density).T * W).T
    return f_eq


def calculate_equilibriums(density, velocity):
    local_velocity_avg = velocity[0, :] ** 2 + velocity[1, :] ** 2
    cu = np.dot(velocity.T, c.T).T
    velocity_2 = cu ** 2
    f_eq = (((1 + 3 * cu + 9 / 2 * velocity_2 - 3 / 2 * local_velocity_avg) * density).T * W).T
    return f_eq


def periodic_boundary_with_pressure_variations(f, rho_in, rho_out):
    # get all the values
    density = calculate_density(f)
    velocity = calculate_velocity(f, density)
    equilibrium = calculate_equilibrium(density, velocity)
    equilibrium_in = calculate_equilibriums(rho_in, velocity[:, -2, :])
    # inlet 1,5,8
    f[:, 0, :] = equilibrium_in + (f[:, -2, :] - equilibrium[:, -2, :])

    # outlet 3,6,7
    equilibrium_out = calculate_equilibriums(rho_out, velocity[:, 1, :])
    # check for correct sizes
    f[:, -1, :] = equilibrium_out + (f[:, 1, :] - equilibrium[:, 1, :])
    return f


def poiseuille_flow():
    # main code
    print("Poiseuille Flow")
    lid_v = 0.000
    steps = 6000
    den_in = den_null + diff
    den_out = den_null - diff

    density = np.ones((x_dim + 2, y_dim + 2))
    velocity = np.zeros((2, x_dim + 2, y_dim + 2))
    f = calculate_equilibrium(density, velocity)

    delta = 2.0 * diff / x_dim / shear_viscosity / 2.
    y = np.linspace(0, y_dim, y_dim + 1) + 0.5
    u_analytical = delta * y * (y_dim - y) / 3.
    x = np.arange(y_dim)
    plt.plot(u_analytical[:-1], x, linestyle='dashed', label='Theoretical')
    plt.legend()

    # loop
    for i in range(steps):
        f = periodic_boundary_with_pressure_variations(f, den_in, den_out)
        f = stream(f)
        f = bounce_back(f, lid_v)
        density = calculate_density(f)
        velocity = calculate_velocity(f, density)
        f = calculate_collision(f, density, velocity, relaxation)
        if i % 100 == 0:
            for i in range(1, 2):
                point = int(i * x_dim / 2)
                plt.plot(velocity[0, point, 1:-1], x)

    print(len(velocity[0, 25, 1:-1]))
    plt.axhline(y=-0.95, color='black', linestyle='dotted', linewidth=3, label='Rigid wall')
    plt.axhline(y=y_dim, color='black', linestyle='dotted', linewidth=3)
    plt.legend()
    plt.xlabel('Position in cross section')
    plt.ylabel('Velocity')
    plt.title('Pouisuelle flow (Gridsize 100x50, Omega = 0.5, delta rho = 0.001)')
    plt.savefig(f"graphs/poiseuli/PouisuelleFlow {steps}.png")
    plt.show()


poiseuille_flow()
