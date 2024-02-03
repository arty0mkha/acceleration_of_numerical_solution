import numpy as np
from scipy.sparse import csc_matrix
from jax import numpy as jnp
from jax import jit
from jax.experimental import sparse
from jax import ops
from functools import partial
from tqdm.notebook import tqdm



def poiseuille_solver(pressure_on_right_boundary, pressure_on_left_boundary, height, length, Ny, viscosity, init):
    """
    Течение Пуазейля, реализация на numpy
    """

    dy = height/Ny
    pressure_gradient = (pressure_on_right_boundary - pressure_on_left_boundary)/length
    const = (dy**2)/viscosity
    vector = np.zeros(shape=Ny)
    vector[1:-1] = const*pressure_gradient
    vector[0] = init[0]
    vector[-1] = init[1]

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


def jax_poiseuille_solver(pressure_on_right_boundary, pressure_on_left_boundary, height, length, Ny, viscosity, init):
    """
    Течение Пуазейля, реализация на jax/numpy
    """

    dy = height/Ny
    pressure_gradient = (pressure_on_right_boundary - pressure_on_left_boundary)/length
    const = (dy**2)/viscosity
    vector = np.zeros(shape=Ny)
    vector[1:-1] = const*pressure_gradient
    vector[0] = init[0]
    vector[-1] = init[1]

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


def u_converge(pressure_on_right_boundary, pressure_on_left_boundary, viscosity, diffusion_constant, vx, height, length, init):
    """
    Правка скорости, для сходимости решения
    """
    dy = 0.8*diffusion_constant/np.max(vx)
    Ny = int(height/dy)+1
    new_vx = jax_poiseuille_solver(pressure_on_right_boundary, pressure_on_left_boundary, height, length, Ny, viscosity, init)
    return new_vx


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


def convection_diffusion_solver(initial_condition, height, length, simulation_time, vx, diffusion_constant):
    """
    Численное решение уравнения конвекции-диффузии на numpy
    """
    Ny = vx.shape[0]
    dy = height/Ny
    dx = dy
    Nx = int(length/dx) + 1
    dx = length/Nx
    dt = dy**2/(4*diffusion_constant)
    Nt = int(simulation_time/dt)
    vx = np.concatenate(([0], vx, [0])) 

    Dx =csc_matrix(make_Dx(Nx+2,dx))
    Dxx =csc_matrix(make_Dxx(Nx+2,dx))
    Dyy =csc_matrix(make_Dyy(Ny+2,dy))

    if Nt>2500:
        Nt = 2500

    concentration = np.zeros(shape=(Nt, Ny+2, Nx+2))
    concentration[0, 1:-1, 1:-1] = initial_condition
    concentration[0, 0] = concentration[0, 1] # boundary bottom
    concentration[0, -1] = concentration[0, -2] # boundary top
    concentration[0, :, 0] = concentration[0, :, 1] # boundary left
    concentration[0, :, -1] = concentration[0, :, -2] # boundary right

    divergence = np.zeros(shape=(vx.shape[0],concentration.shape[-1]))
    for i in tqdm(range(Nt-1)):
        laplace = Dyy@concentration[i] + concentration[i]@Dxx
        dconcentration_dx = concentration[i]@Dx
        for j in range(1,Ny):
            divergence[j] = vx[j]*dconcentration_dx[j]
        concentration[i+1] = concentration[i] + dt*(diffusion_constant*laplace - divergence)
    concentration[i+1, 0] = concentration[i+1, 1] # boundary bottom
    concentration[i+1, -1] = concentration[i+1, -2] # boundary top
    concentration[i+1, :, 0] = concentration[i+1, :, 1] # boundary left
    concentration[i+1, :, -1] = concentration[i+1, :, -2] # boundary right

    return concentration[:, 1:-1, 1:-1], dt


def jax_convection_diffusion_solver(initial_condition, height, length, simulation_time, vx, diffusion_constant):
    """
    Численное решение уравнения конвекции-диффузии, полная хр нь
    """

    def make_matrix(shape):
        matrix = np.eye(shape)
        matrix[0,0] = 0
        matrix[0,1] = 1
        matrix[-1,-1] = 0
        matrix [-1, -2] = 1
        return matrix


    Ny = vx.shape[0]
    dy = height/Ny
    dx = dy
    Nx = int(length/dx) + 1
    dx = length/Nx
    dt = (dy**2/(4*diffusion_constant))
    Nt = int(simulation_time/dt)
    vx = np.concatenate(([0], vx, [0])) 


    @partial(jit, static_argnums=(2))
    def sp_matmul(A, B, shape):
        assert B.ndim == 2
        indexes, values = A
        rows, cols = indexes
        in_ = B.take(cols, axis=0)
        prod = in_*values[:, None]
        res = ops.segment_sum(prod, rows, shape)
        return res


    @partial(jit, static_argnums=(2))
    def sp_matmul_reverse(A,B, shape):
        assert A.ndim == 2
        indexes, values = B
        rows, cols = indexes
        in_ = A.take(cols, axis=0)
        prod = in_*values[:, None]
        res = ops.segment_sum(prod, rows, shape)
        return res

    Dx = sparse.BCOO.fromdense(make_Dx(Nx+2,dx))
    Dxx = sparse.BCOO.fromdense(make_Dxx(Nx+2,dx))
    Dyy = sparse.BCOO.fromdense(make_Dyy(Ny+2,dy))

    Sx = sparse.BCOO.fromdense(make_matrix(Nx+2))
    Sy = sparse.BCOO.fromdense(make_matrix(Ny+2).T)

    concentration = jnp.zeros(shape=(Nt, Ny+2, Nx+2))
    concentration = concentration.at[0, 1:-1, 1:-1].set(initial_condition)
    val =  sp_matmul_reverse(sp_matmul((Sy.indices.T, Sy.data), concentration[0], Ny+2), (Sx.indices.T, Sx.data), Ny+2)
    concentration = concentration.at[0].set(val)

    divergence = np.zeros(shape=(vx.shape[0],concentration.shape[-1]))
    for i in tqdm(range(Nt-1)):
        first = sp_matmul((Dyy.indices.T, Dyy.data), concentration[i], Ny+2)
        second = sp_matmul_reverse(concentration[i], (Dxx.indices.T, Dxx.data), Ny+2)
        laplace = first + second
        dconcentration_dx = sp_matmul_reverse(concentration[i], (Dx.indices.T, Dx.data), Ny+2)
        for j in range(1,Ny):
            divergence[j] = vx[j]*dconcentration_dx[j]
        concentration = concentration.at[i+1].set(concentration[i] + dt*(diffusion_constant*laplace - divergence))
        matr = sp_matmul((Sy.indices.T, Sy.data), concentration[i+1], Ny+2)
        concentration = concentration.at[i+1].set(sp_matmul_reverse(matr, (Sx.indices.T, Sx.data), Ny+2))
    return concentration[:, 1:-1, 1:-1], dt
