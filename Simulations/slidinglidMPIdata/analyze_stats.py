import numpy as np
import matplotlib.pyplot as plt

c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
c = np.vstack((c_x, c_y)).T


f=np.load('final_f.npy')
sizef=np.shape(f)[1]

def calculate_density(f):
    return np.sum(f, axis=0)

def calculate_velocity(f, density):
    velocity = (np.dot(f.T, c).T / density)
    return velocity


density= calculate_density(f)
velocity= calculate_velocity(f,density)

def v_plot(velocity):
    """Calculates Velocity profile"""
    v = np.sqrt(velocity[0, :, :]**2 + velocity[1, :, :]**2)
    #print(np.shape(v))
    return v

def plot_fig():
    X, Y = np.meshgrid(np.arange(np.shape(f)[1]), np.arange(np.shape(f)[1]))
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
        'title': "Sliding Lid (Gridsize {}x{})".format(np.shape(f)[1], np.shape(f)[1]),
        'xlabel': "x-Position",
        'ylabel': "y-Position",
    }

    # Plot using the unpacking operator
    plt.streamplot(X, Y, velocity[0, :, :].T, velocity[1, :, :].T, color=v_mag.T)
    plt.title(plot_properties['title'])
    plt.xlabel(plot_properties['xlabel'])
    plt.ylabel(plot_properties['ylabel'])
    fig = plt.colorbar()
    fig.set_label("Velocity", rotation=270, labelpad=15)
    plt.savefig('temp_r.png')
    plt.show()

plot_fig()