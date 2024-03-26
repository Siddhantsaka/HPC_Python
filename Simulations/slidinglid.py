import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


rn = 1000
steps = 100000

sizef = 300
x_dim = 302
y_dim = 302
c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
c = np.vstack((c_x, c_y)).T
lid_v = 0.1
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]).T
relaxation = (2*rn)/(6 * sizef * lid_v + rn)
print(relaxation)

def calculate_velocity(f, density):
    velocity = (np.dot(f.T, c).T / density)
    return velocity


def calculate_density(f):
    return np.sum(f, axis=0)


def stream(f):
    for i in range(1,9):
        f[i] = np.roll(f[i],c[i], axis=(0, 1))
    return f


def calculate_equilibrium(density, velocity):
    local_velocity_avg = velocity[0, :, :] ** 2 + velocity[1, :, :] ** 2
    cu = np.dot(velocity.T, c.T).T
    velocity_2 = cu ** 2
    f_eq = (((1 + 3 * cu + 9 / 2 * velocity_2 - 3 / 2 * local_velocity_avg) * density).T * W).T
    return f_eq


def calculate_collision(f, density, velocity, rl):
    f_eq = calculate_equilibrium(density, velocity)
    f -= rl * (f-f_eq)
    return f


def bounce_back(f,lid_v):
    # Bottom wall
    f[[1, 5, 8], 1, 1:-1] = f[[3, 7, 6], 0, 1:-1]
    # Top wall
    f[[3, 6, 7], -2, 1:-1] = f[[1, 8, 5], -1, 1:-1]
    # Left and right wall
    f[[2, 5, 6], 1:-1, 1] = f[[4, 7, 8], 1:-1, 0]

    # Adjustments for lid-driven cavity flow at the top wall
    d_w = 2.0 * f[[2, 5, 6], 1:-1, -1].sum(axis=0) + f[[0, 1, 3], 1:-1, -1].sum(axis=0)
    f[4, 1:-1, -2] = f[2, 1:-1, -1]
    f[7, 1:-1, -2] = f[5, 1:-1, -1] - lid_v * d_w / 6
    f[8, 1:-1, -2] = f[6, 1:-1, -1] + lid_v * d_w / 6
    return f

def v_plot(velocity):
    v = np.sqrt(velocity[0, 1:-1, 1:-1]**2 + velocity[1, 1:-1, 1:-1]**2)
    return v


def sliding_lid():
    print('Sliding Lid Simulation')

    density = np.ones((x_dim+2, y_dim+2))
    velocity = np.zeros((2, x_dim+2, y_dim+2))
    f = calculate_equilibrium(density,velocity)
    plot_every = 10000

    def plot_fig(b):
        X, Y = np.meshgrid(np.arange(y_dim), np.arange(x_dim))
        v_mag = v_plot(velocity)
        print(np.shape(v_mag))
        print(np.shape(velocity))

        plot_properties = {
            'X': X,
            'Y': Y,
            'U': velocity[0, 1:-1, 1:-1].T,
            'V': velocity[1, 1:-1, 1:-1].T,
            'color': v_mag.T,
            'xlim': [0, sizef + 1],
            'ylim': [0, sizef + 1],
            'title': "Sliding Lid (Gridsize {}x{}, omega = {:.2f}, steps = {})".format(x_dim, y_dim, relaxation, b),
            'xlabel': "x-Position",
            'ylabel': "y-Position",
        }

        # Plot using the unpacking operator
        plt.clf()
        plt.streamplot(X, Y, velocity[0, 1:-1, 1:-1].T, velocity[1, 1:-1, 1:-1].T, color=v_mag.T)
        plt.title(plot_properties['title'])
        plt.xlabel(plot_properties['xlabel'])
        plt.ylabel(plot_properties['ylabel'])
        fig = plt.colorbar()
        fig.set_label("Velocity", rotation=270, labelpad=15)
        plt.savefig(f'slidingLidGraphs/temp_r {b}.png')


    for i in range(steps):
        f = stream(f)
        f = bounce_back(f, lid_v)
        density = calculate_density(f)
        velocity = calculate_velocity(f, density)
        f = calculate_collision(f, density, velocity, relaxation)
        print(f"Progress: {i}", end='\r', flush=True)
        if (i % plot_every) == 0:
            plot_every = plot_every*2
            plot_fig(i)



    plot_fig(steps)

def sliding_lidAni():

    def update_plot(frame):
        global f, density, velocity

        # Calculate the new state of the simulation for each frame
        f = stream(f)
        f = bounce_back(f, lid_v)
        density = calculate_density(f)
        velocity = calculate_velocity(f, density)
        f = calculate_collision(f, density, velocity, relaxation)
        print(f"Progress: {frame}", end='\r', flush=True)

        # Update the plot
        if frame % 10 == 0:
            plt.clf()
            X, Y = np.meshgrid(np.arange(y_dim), np.arange(x_dim))
            v_mag = v_plot(velocity)
            plt.streamplot(X, Y, velocity[0, 1:-1, 1:-1].T, velocity[1, 1:-1, 1:-1].T, color=v_mag.T)
            plt.title("Sliding Lid with lid v = 0.1")
            plt.xlabel("x-Position")
            plt.ylabel("y-Position")
            fig = plt.colorbar()
            fig.set_label("Velocity", rotation=270, labelpad=15)
    rn = 1000
    steps = 100000
    sizef = 300
    x_dim = 302
    y_dim = 302
    c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    c = np.vstack((c_x, c_y)).T
    lid_v = 0.1
    W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]).T
    relaxation = (2*rn)/(6 * sizef * lid_v + rn)

    density = np.ones((x_dim+2, y_dim+2))
    velocity = np.zeros((2, x_dim+2, y_dim+2))
    global f
    f = calculate_equilibrium(density, velocity)

    # Create the animation using FuncAnimation
    animation = FuncAnimation(plt.gcf(), update_plot, frames=steps, interval=10, repeat=False)
    animation.save("sliding_lid_animation2.gif", writer="pillow")

    # Show the plot
    #plt.show()


#sliding_lidAni()
sliding_lid()
