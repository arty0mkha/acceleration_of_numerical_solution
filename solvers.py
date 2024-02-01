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


def jax_poiseuille_solver(pressure_on_right_boundary, pressure_on_left_boundary, height, length, Ny, viscosity):
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
    
    def second_step(matrix, vector):
        return jnp.matmul(jnp.linalg.inv(matrix), vector)
    return second_step(matrix, vector)


def convection_diffusion_solver(initial_condition, height, length, simulation_time, vx, diffusion_constant):

    def make_Dx(N,dx):
        Dx = np.zeros(shape=(N, N))
        for i in range(N-1):
            Dx[i, i] = -1
            Dx[i+1, i] = 1
        Dx /= dx
        return Dx 
    
    def make_Dxx(N,dx):
        Dxx = np.zeros(shape=(N, N))
        for i in range(1, N-1):
            Dxx[i-1, i] = 1
            Dxx[i, i] = -2
            Dxx[i+1, i] = 1
        Dxx /= dx**2
        return Dxx 
    
    def make_Dyy(N,dy):
        Dyy = np.zeros(shape=(N, N))
        for i in range(1, N-1):
            Dyy[i, i-1] = 1
            Dyy[i, i] = -2
            Dyy[i, i+1] = 1
        Dyy /= dy**2
        return Dyy
    
    Ny = vx.shape[0]
    Nx = int(length/height*Ny)
    dx = length/Nx
    dy = height/Ny
    dt = (dx**2 + dy**2)/8
    Nt = int(simulation_time/dt)

    vx = np.concatenate( ([0], vx, [0])) 

    Dx =csc_matrix(make_Dx(Nx+2,dx))
    Dxx =csc_matrix(make_Dxx(Nx+2,dx))
    Dyy =csc_matrix(make_Dyy(Ny+2,dy))

    # neumann conditions on all boundaries?
    concentration = np.zeros(shape=(Nt, Ny+2, Nx+2))
    concentration[0, 1:-1, 1:-1] = initial_condition
    concentration[0, 0, 1:-1] = concentration[0, 1, 1:-1] # boundary top
    concentration[0, -1, 1:-1] = concentration[0, -2, 1:-1] # boundary bottom
    concentration[0, 0, 1:-1] = concentration[0, 1, 1:-1] # boundary left
    concentration[0, -1, 1:-1] = concentration[0, -2, 1:-1] # boundary right

    divergence = np.zeros(shape=(vx.shape[0],concentration.shape[-1]))
    for i in range(Nt-1):
        laplace = Dyy@concentration[i] + concentration[i]@Dxx
        dconcentration_dx = concentration[i]@Dx
        for j in range(1,Ny):
            divergence[j] = vx[j]*dconcentration_dx[j]
        concentration[i+1] = concentration[i] + dt*(diffusion_constant*laplace - divergence)
        concentration[i+1, 0, 1:-1] = concentration[i+1, 1, 1:-1] # boundary top
        concentration[i+1, -1, 1:-1] = concentration[i+1, -2, 1:-1] # boundary bottom
        concentration[i+1, 0, 1:-1] = concentration[i+1, 1, 1:-1] # boundary left
        concentration[i+1, -1, 1:-1] = concentration[i+1, -2, 1:-1] # boundary right

    return concentration, dt


def jax_convection_diffusion_solver(initial_condition, height, length, simulation_time, vx, diffusion_constant):
    
    def make_Dx(N,dx):
        Dx = np.zeros(shape=(N, N))
        for i in range(N-1):
            Dx[i, i] = -1
            Dx[i+1, i] = 1
        Dx /= dx
        return Dx 


    def make_Dxx(N,dx):
        Dxx = np.zeros(shape=(N, N))
        for i in range(1, N-1):
            Dxx[i-1, i] = 1
            Dxx[i, i] = -2
            Dxx[i+1, i] = 1
        Dxx /= dx**2
        return Dxx 


    def make_Dyy(N,dy):
        Dyy = np.zeros(shape=(N, N))
        for i in range(1, N-1):
            Dyy[i, i-1] = 1
            Dyy[i, i] = -2
            Dyy[i, i+1] = 1
        Dyy /= dy**2
        return Dyy


    Ny = vx.shape[0]
    Nx = int(length/height*Ny)
    dx = length/Nx
    dy = height/Ny
    dt = (dx**2 + dy**2)/8
    Nt = int(simulation_time/dt)

    vx = np.concatenate( ([0], vx, [0])) 

    Dx = make_Dx(Nx+2,dx)
    Dxx = make_Dxx(Nx+2,dx)
    Dyy = make_Dyy(Ny+2,dy)

    Dx = sparse.BCOO.fromdense(Dx)
    Dxx = sparse.BCOO.fromdense(Dxx)
    Dyy = sparse.BCOO.fromdense(Dyy)

    # neumann conditions on all boundaries?
    concentration = np.zeros(shape=(Nt, Ny+2, Nx+2))
    concentration[0, 1:-1, 1:-1] = initial_condition
    concentration[0, 0, 1:-1] = concentration[0, 1, 1:-1] # boundary top
    concentration[0, -1, 1:-1] = concentration[0, -2, 1:-1] # boundary bottom
    concentration[0, 0, 1:-1] = concentration[0, 1, 1:-1] # boundary left
    concentration[0, -1, 1:-1] = concentration[0, -2, 1:-1] # boundary right

    divergence = np.zeros(shape=(vx.shape[0],concentration.shape[-1]))

    @partial(jit, static_argnums=(2))
    def sp_matmul(A, B, shape):
        """
        Arguments:
            A: (N, M) sparse matrix represented as a tuple (indexes, values)
            B: (M,K) dense matrix
            shape: value of N
        Returns:
            (N, K) dense matrix
        """
        assert B.ndim == 2
        indexes, values = A
        rows, cols = indexes
        in_ = B.take(cols, axis=0)
        prod = in_*values[:, None]
        res = ops.segment_sum(prod, rows, shape)
        return res


    for i in range(Nt-1):
        laplace = sp_matmul((Dyy.indices.T, Dyy.data), concentration[i], Ny+2) + sp_matmul((Dxx.indices.T, Dxx.data), concentration[i], concentration[i].shape[0])
        dconcentration_dx = sp_matmul((Dx.indices.T, Dx.data), concentration[i], concentration[i].shape[0])
        for j in range(1,Ny):
            divergence[j] = vx[j]*dconcentration_dx[j]
        concentration[i+1] = concentration[i] + dt*(diffusion_constant*laplace - divergence)
        concentration[i+1, 0, 1:-1] = concentration[i+1, 1, 1:-1] # boundary top
        concentration[i+1, -1, 1:-1] = concentration[i+1, -2, 1:-1] # boundary bottom
        concentration[i+1, 0, 1:-1] = concentration[i+1, 1, 1:-1] # boundary left
        concentration[i+1, -1, 1:-1] = concentration[i+1, -2, 1:-1] # boundary right
    return concentration, dt
