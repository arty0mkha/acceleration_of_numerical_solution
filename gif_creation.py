import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def anim_result(data_u:np.ndarray, timestep, L, H, picture:bool, showMe:bool):
    """
    Make animation of u(t,x,y) and V(t,x,y) and save it to gif
    """


    def animate(i):                                                                                     # animate function for animation.FuncAnimation
        line.set_array(data_u[i])                                                                       # change "slice" of u cube
        ax.set_title(f'concentration field at t = {round(i*timestep,4)}')                                               # also change quiver 
        return line
    
    fig, ax = plt.subplots()
    line = plt.imshow(data_u[0], aspect = 'auto', cmap = 'turbo', extent = [0,L,0,H])             # u slice
    plt.colorbar(line, ax=ax)                                                                   
        # gradient for V

    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_title('concentration field at t = {t}'.format(t=0))

    ani = animation.FuncAnimation(fig, animate, interval=timestep*100, blit=False, frames = data_u.shape[0])       # animate


    if showMe == True:
        plt.show()

    if picture == True:
        writer = animation.PillowWriter(                                                                # saving picture
            fps=30, metadata=dict(artist='Doofenshmirtz Evil Incorporated'), bitrate=1800)
        ani.save(f"gifs/t={round(data_u.shape[0]*timestep,3)}, L={L}, H={H}.gif", writer=writer)
