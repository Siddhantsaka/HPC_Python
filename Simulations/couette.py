import numpy as np
import matplotlib.pyplot as plt

x_dim = 100
y_dim = 50
c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
c = np.vstack((c_x, c_y)).T
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]).T
relaxation = 0.5


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


def bounce_back(f, lid_v):
    f[[2, 5, 6], 1:-1, 1] = f[[4, 7, 8], 1:-1, 0]
    f[4, 1:-1, -2] = f[2, 1:-1, -1]
    f[7, 1:-1, -2] = f[5, 1:-1, -1] - 1 / 6 * lid_v
    f[8, 1:-1, -2] = f[6, 1:-1, -1] + 1 / 6 * lid_v
    return f


def calculate_equilibrium(density, velocity):
    local_velocity_avg = velocity[0, :, :] ** 2 + velocity[1, :, :] ** 2
    cu = np.dot(velocity.T, c.T).T
    velocity_2 = cu ** 2
    f_eq = (((1 + 3 * cu + 9 / 2 * velocity_2 - 3 / 2 * local_velocity_avg) * density).T * W).T
    return f_eq


def couette_flow():
    print("couette Flow")
    steps = 4000
    plot_every = 20
    lid_v = 0.1
    density = np.ones((x_dim, y_dim + 2))
    velocity = np.zeros((2, x_dim, y_dim + 2))
    f = calculate_equilibrium(density, velocity)
    print(f.shape)
    x = np.arange(0, y_dim)

    for i in range(steps):
        density = calculate_density(f)
        velocity = calculate_velocity(f, density)
        f = calculate_collision(f, density, velocity, relaxation)
        f = stream(f)
        f = bounce_back(f, lid_v)
        print(i, f'of {steps}')

        if (i % plot_every) == 0:
            if i > 0:
                plot_every = plot_every * 2
                plt.plot(velocity[0, 50, 1:-1], x)
                if i == 200:
                    plt.plot(velocity[0, 50, 1:-1], x, label="Simulated".format(i))

    plt.axhline(y=-0.95, color='black', linestyle='dotted', linewidth=3, label='Rigid wall')
    plt.axhline(y=y_dim, color='red', linestyle='dotted', linewidth=3, label='Moving wall')

    x = np.arange(0, y_dim)
    y = lid_v * 1 / 50 * x
    plt.plot(y, x, linewidth=2, label="Analytical")
    plt.legend()
    plt.xlabel('Velocity')
    plt.ylabel('Cross section position')
    plt.title(f'Couette flow (Gridsize 100x50, Iterations {steps})')
    savestring = "CouetteFlow.png"
    plt.savefig(savestring)
    plt.show()


couette_flow()
