import numpy as np
from tqdm import tqdm
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