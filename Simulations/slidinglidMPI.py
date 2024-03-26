import numpy as np
from mpi4py import MPI
#import matplotlib.pyplot as plt
import psutil
import time
import sys

# Only vars

cores = psutil.cpu_count(logical=False)
c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
c = np.vstack((c_x, c_y)).T
comm = MPI.COMM_WORLD
st = time.time()
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]).T


mpi_s = {
    "boundary": {key: False for key in ["left", "right", "top", "bottom"]},
    "neighbours": {key: -1 for key in ["left", "right", "top", "bottom"]},
    **{key: -1 for key in ["rank", "size", "pos_x", "pos_y", "size_x", "size_y", "relaxation", "dim_f", "steps", "lid_v"]}
}


def update(rank, size, mx, my, dim_f, relaxation, steps, lid_v):
    mpi_s.update({
        **{key: value for key, value in zip(["rank", "dim_f", "size"], [rank, dim_f, size])},
        **{key: value for key, value in zip(["relaxation", "steps", "lid_v"], [relaxation, steps, lid_v])},
        **dict(zip(["size_x", "size_y"], [int(dim_f // mx + 2), int(dim_f // my + 2)])),
        **dict(zip(["pos_x", "pos_y"], positions(rank, size))),
        "boundary": b_assign(*positions(rank, size), mx - 1, my - 1),
        "neighbours": f_ne(rank, size)
    })


def b_assign(px, py, mx, my):
    return {
        "left": px == 0,
        "right": px == mx,
        "top": py == my,
        "bottom": py == 0
    }


def positions(rank, size):
    py, px = divmod(rank, int(np.sqrt(size)))
    return px, py


def f_ne(rank, size):
    return {"left": rank - 1, "right": rank + 1, "top": rank + int(size ** 0.5), "bottom": rank - int(size ** 0.5)}


def stream(f):
    for i in range(1,9):
        f[i] = np.roll(f[i],c[i], axis=(0, 1))
    return f

def calculate_velocity(f, density):
    velocity = (np.dot(f.T, c).T / density)
    return velocity

def calculate_density(f):
    return np.sum(f, axis=0)

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

def v_plot(velocity):
    v = np.sqrt(velocity[0, :, :]**2 + velocity[1, :, :]**2)

    return v


def MPI_bounceback(f):
    if mpi_s["boundary"]["right"]:
        f[[3, 6, 7], -2, :] = f[[1, 8, 5], -1, :]  # Update "right" boundary

    if mpi_s["boundary"]["left"]:
        f[[1, 5, 8], 1, :] = f[[3, 7, 6], 0, :]    # Update "left" boundary

    if mpi_s["boundary"]["bottom"]:
        f[2, :, 1] = f[4, :, 0]                   # Update "bottom" boundary
        f[[5, 6], :, 1] = f[[7, 8], :, 0]         # Update "bottom" boundary for specified directions

    if mpi_s["boundary"]["top"]:
        f[4, :, -2] = f[2, :, -1]                 # Update "top" boundary
        f[[7, 8], :, -2] = [f[5, :, -1] - 1 / 6 * mpi_s["lid_v"], f[6, :, -1] + 1 / 6 * mpi_s["lid_v"]]  # Update "top" boundary for specified directions

    return f


def comunicate(f, comm):
    if not mpi_s["boundary"]["right"]:
        rb = f[:, -1, :].copy()
        comm.Sendrecv(f[:, -2, :].copy(), mpi_s["neighbours"]["right"], recvbuf=rb, sendtag=11, recvtag=12)
        f[:, -1, :] = rb

    if not mpi_s["boundary"]["left"]:
        rb = f[:, 0, :].copy()
        comm.Sendrecv(f[:, 1, :].copy(), mpi_s["neighbours"]["left"], recvbuf=rb, sendtag=12, recvtag=11)
        f[:, 0, :] = rb

    if not mpi_s["boundary"]["bottom"]:
        rb = f[:, :, 0].copy()
        comm.Sendrecv(f[:, :, 1].copy(), mpi_s["neighbours"]["bottom"], recvbuf=rb, sendtag=99, recvtag=98)
        f[:, :, 0] = rb

    if not mpi_s["boundary"]["top"]:
        rb = f[:, :, -1].copy()
        comm.Sendrecv(f[:, :, -2].copy(), mpi_s["neighbours"]["top"], recvbuf=rb, sendtag=98, recvtag=99)
        f[:, :, -1] = rb

    return f


def combine(f, comm):
    ff = np.ones((9, mpi_s["dim_f"], mpi_s["dim_f"])) if mpi_s["rank"] == 0 else None
    ox = mpi_s["size_x"] - 2
    oy = mpi_s["size_y"] - 2

    if mpi_s["rank"] == 0:
        ff[:, 0:ox, 0:oy] = f[:, 2:-2, 2:-2]

    temp = np.empty((9, ox, oy))

    for i in range(1, mpi_s["size"]):
        if mpi_s["rank"] == 0:
            comm.Recv(temp, source=i, tag=i)
            x, y = positions(i, mpi_s["size"])
            csx = ox * x
            cex = ox * (x + 1)
            csy = oy * y
            cey = oy * (y + 1)
            ff[:, csx:cex, csy:cey] = temp
        elif mpi_s["rank"] == i:
            comm.Send(f[:, 2:-2, 2:-2].copy(), dest=0, tag=i)

    return ff



def sliding_lid_mpi(comm):
    density = np.ones((mpi_s["size_x"] + 2, mpi_s["size_y"] + 2),dtype= np.float32)
    velocity = np.zeros((2, mpi_s["size_x"] + 2, mpi_s["size_y"] + 2),dtype= np.float32)
    f = calculate_equilibrium(density, velocity)

    for i in range(mpi_s["steps"]):
        f = stream(f)
        f = MPI_bounceback(f)
        density = calculate_density(f)
        velocity = calculate_velocity(f, density)
        f = calculate_collision(f, density, velocity, mpi_s["relaxation"])
        comunicate(f, comm)

    f_com = combine(f,comm)
    plotter(f_com)

def plotter(f_com):
    #plot
    if mpi_s["rank"] == 0:


        np.save('final_f.npy',f_com)
        # recalculate ux and uy
        savestring = "slidingLidmpi " + str(mpi_s["size"]) + ".txt"
        f = open(savestring, "w")
        tt = time.time() - float(st)
        f.write(str(tt))
        f.close()



def call():
    if len(sys.argv) != 3:
        print("Usage: python testMpiC.py <steps> <f_len>")
        return

    #steps = 500
    steps = int(sys.argv[1])
    re = 1000
    #f_len = 300
    f_len = int(sys.argv[2])
    lid_v = 0.1
    relaxation = (2 * re) / (6 * f_len * lid_v + re)
    update(rank=comm.Get_rank(), size=comm.Get_size(), mx=np.sqrt(comm.Get_size()), my=np.sqrt(comm.Get_size()), dim_f=f_len, relaxation=relaxation, steps=steps, lid_v=lid_v)
    print(mpi_s)

    print()
    sliding_lid_mpi(comm)

call()
