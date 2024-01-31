import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import jax.numpy as jnp
from jax import jit
from jax import lax


def poiseuille_solver(pressure_on_right_boundary, pressure_on_left_boundary, height, length, Ny, viscosity):
    # dirichlet conditions u(0)=u(h)=0
    dy = height/Ny
    pressure_gradient = (pressure_on_right_boundary - pressure_on_left_boundary)/length
    const = (dy**2)/viscosity
    vector = np.zeros(shape=Ny)
    vector[1:-1] = const*pressure_gradient

    matrix = np.zeros(shape=(Ny,Ny))
    matrix[0, 0] = 1
    matrix[-1, -1] = 1
    for i in range(1,Ny-1):
        matrix[i, i-1] = 1
        matrix[i, i] = -2
        matrix[i, i+1] = 1

    inverse_matrix = np.linalg.inv(matrix)
    velocity = np.matmul(inverse_matrix, vector)
    return velocity


def jax_solver(pressure_on_right_boundary, pressure_on_left_boundary, height, length, Ny, viscosity):
    def jax_change_element(matrix, pos1, pos2, value):
        return matrix.at[pos1].set(matrix[pos1].at[pos2].set(value))

    def loopfunction (matrix, i):
        matrix = jitchange(matrix, i, i-1, 1.)
        matrix = jitchange(matrix, i, i, -2.)
        matrix = jitchange(matrix, i, i+1, 1.)
        return matrix

    jitchange = jit(jax_change_element)
    jitloop = jit(loopfunction)

    dy = jnp.divide(height, Ny)

    pressure_gradient = jnp.divide(jnp.add(pressure_on_right_boundary, -pressure_on_left_boundary), length)
    const = jnp.divide(jnp.dot(dy, dy), viscosity)

    val = jnp.dot(const, pressure_gradient)
    vector = jnp.concatenate((jnp.array([0]), val*jnp.ones(Ny - 2), jnp.array([0])))
    matrix = jnp.zeros(shape=(Ny,Ny))

    matrix = jitchange(matrix, 0, 0, 1.)
    matrix = jitchange(matrix, -1, -1, 1.)
    matrix = lax.fori_loop(1, Ny-1, lambda i,matrix_local: jitloop(matrix_local, i), matrix)

    return jnp.matmul(jnp.linalg.inv(matrix), vector)

jax_poiseuille_solver = jit(jax_solver, static_argnums=(4,))


def convection_diffusion_solver(initial_condition, height, length, simulation_time, vx, diffusion_constant):
    
    Ny = vx.shape[0]
    Nx = int(length/height*Ny)
    dx = length/Nx
    dy = height/Ny
    dt = (dx**2 + dy**2)/8
    Nt = int(simulation_time/dt)
    characteristic_length = dx*dy/(2*dx + 2*dy)
    Pe = characteristic_length*vx/diffusion_constant

    vx = np.concatenate( ([0], vx, [0]))
    if np.max(Pe) > 0:
        new_initial_condition = np.zeros(initial_condition.shape)
        for j in range(initial_condition.shape[1]):
            new_initial_condition [:, -j-1] = initial_condition[:, j]
        initial_condition = new_initial_condition
        vx = -vx 
        print('revert')     
    
    # neumann conditions on all boundaries?
    concentration = np.zeros(shape=(Nt, Ny+2, Nx+2))
    concentration[0, 1:-1, 1:-1] = initial_condition

    concentration[0, 0, 1:-1] = concentration[0, 1, 1:-1] # boundary top
    concentration[0, -1, 1:-1] = concentration[0, -2, 1:-1] # boundary bottom
    concentration[0, 0, 1:-1] = concentration[0, 1, 1:-1] # boundary left
    concentration[0, -1, 1:-1] = concentration[0, -2, 1:-1] # boundary right

    Dx = np.zeros(shape=(Nx+2, Nx+2))
    for i in range(Nx+1):
        Dx[i, i] = -1
        Dx[i+1, i] = 1
    Dx /= dx

    Dxx = np.zeros(shape=(Nx+2, Nx+2))
    for i in range(1, Nx+1):
        Dxx[i-1, i] = 1
        Dxx[i, i] = -2
        Dxx[i+1, i] = 1
    Dxx /= dx**2

    Dyy = np.zeros(shape=(Ny+2, Ny+2))
    for i in range(1, Ny+1):
        Dyy[i, i-1] = 1
        Dyy[i, i] = -2
        Dyy[i, i+1] = 1
    Dyy /= dy**2 

    divergence = np.zeros(shape=(vx.shape[0],concentration.shape[-1]))
    for i in tqdm(range(Nt-1)):
        laplace = np.matmul(Dyy,concentration[i]) + np.matmul(concentration[i], Dxx)
        dconcentration_dx = np.matmul(concentration[i], Dx)
        for j in range(1,Ny):
            divergence[j] = vx[j]*dconcentration_dx[j]
        concentration[i+1] = concentration[i] + dt*(diffusion_constant*laplace - divergence)
        concentration[i+1, 0, 1:-1] = concentration[i+1, 1, 1:-1] # boundary top
        concentration[i+1, -1, 1:-1] = concentration[i+1, -2, 1:-1] # boundary bottom
        concentration[i+1, 0, 1:-1] = concentration[i+1, 1, 1:-1] # boundary left
        concentration[i+1, -1, 1:-1] = concentration[i+1, -2, 1:-1] # boundary right
    
    if  np.max(Pe) > 0:
        new_concentration = np.zeros(concentration.shape)
        for j in range(initial_condition.shape[-1]):
            new_concentration[:, :, -j-1] = concentration[:, :, j] 
        concentration = new_concentration

    return concentration, dt, Pe


def jax_convection_diffusion(initial_condition, height, length, simulation_time, vx, diffusion_constant):

    def loopfun(initial_condition, vx):
        new_initial_condition = jnp.zeros(initial_condition.shape[0])
        for j in range(initial_condition.shape[1]):
            new_initial_condition = initial_condition.at[:, jnp.add(-j, -1)].set(initial_condition[:, j])
        initial_condition = new_initial_condition
        vx = -vx 
        print('revert')   
        return True
    
    Ny = vx.shape[0]
    Nx = int(length/height*Ny)

    dx = length/Nx
    dy = height/Ny
    dt = (dx**2 + dy**2)/8
    Nt = int(simulation_time/dt)

    characteristic_length = jnp.divide(jnp.dot(dx,dy), jnp.add(jnp.dot(2, dx), jnp.add(2, dy)))
    Pe = jnp.divide(jnp.dot(characteristic_length, vx), diffusion_constant)
    #lax.batch_matmul
    vx = jnp.concatenate((jnp.array([0]), vx, jnp.array([0])))

    #lax.cond(pred = lax.gt(Pe[jnp.int64(jnp.divide(Pe.shape[0],2))], 0), loopfun(initial_condition, vx), false_fun = None)

    # neumann conditions on all boundaries?
    print (type(Nx))
    print (type(Ny))
    concentration = jnp.zeros(shape = (Nt, Ny+2, Nx+2))

    concentration = concentration.at[0, 1:-1, 1:-1].set(initial_condition)

    concentration = concentration.at[0, 0, 1:-1].set(concentration[0, 1, 1:-1])     # boundary top
    concentration = concentration.at[0, -1, 1:-1].set(concentration[0, -2, 1:-1])   # boundary bottom
    concentration = concentration.at[0, 0, 1:-1].set(concentration[0, 1, 1:-1])     # boundary left
    concentration = concentration.at[0, -1, 1:-1].set(concentration[0,-2,1:-1])     # boundary right

    Dx = jnp.zeros(shape=(Nx+2, Nx+2))
    for i in range(Nx+1):
        Dx = Dx.at[i, i].set(-1.)
        Dx = Dx.at[i+1, i].set(1.)
    Dx = jnp.divide(Dx, dx)

    Dxx = jnp.zeros(shape=(Nx+2, Nx+2))
    for i in range(1, Nx+1):
        Dxx = Dxx.at[jnp.add(i, -1), i].set(1.)
        Dxx = Dxx.at[i, i].set(-2.)
        Dxx = Dxx.at[jnp.add(i, 1), i].set(1.)

    Dxx = jnp.divide(Dxx, jnp.dot(dx, dx))

    Dyy = jnp.zeros(shape=(Ny+2, Ny+2))
    for i in range(1, Ny+1):
        Dyy = Dyy.at[i, jnp.add(i, -1)].set(1.)
        Dyy = Dyy.at[i, i].set(-2.)
        Dyy = Dyy.at[i, jnp.add(i, 1)].set(1.)

    Dyy = jnp.divide(Dyy, jnp.dot(dy, dy))

    def add(x,y):
        return x + y
    
    divergence = jnp.zeros(shape=(vx.shape[0], concentration.shape[-1]))
    for i in range(Nt-1):
        laplace = vmap(add)(jnp.matmul(Dyy,concentration[i]), jnp.matmul(concentration[i], Dxx))
        dconcentration_dx = jnp.matmul(concentration[i], Dx)
        for j in range(1,Ny):
            divergence = divergence.at[j].set(jnp.dot(vx[j], dconcentration_dx[j]))
        concentration = concentration.at[i+1].set(jnp.add(concentration[i], jnp.dot(dt, jnp.add(jnp.dot(diffusion_constant,laplace), divergence))))

        concentration = concentration.at[jnp.add(i, 1), 0, 1:-1].set(concentration[jnp.add(i, 1), 1, 1:-1])     # boundary top
        concentration = concentration.at[jnp.add(i, 1), -1, 1:-1].set(concentration[jnp.add(i, 1), -2, 1:-1])   # boundary bottom
        concentration = concentration.at[jnp.add(i, 1), 0, 1:-1].set(concentration[jnp.add(i, 1), 1, 1:-1])     # boundary left
        concentration = concentration.at[jnp.add(i, 1), -1, 1:-1].set(concentration[jnp.add(i, 1), -2, 1:-1])   # boundary right
        Pe = Pe + 1
    print ('aaaaa')

    return concentration, dt, Pe
