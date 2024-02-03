import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageOps



def anim_result(data_u:np.ndarray, timestep:float, L:float, H:float, picture:bool, showMe:bool, colour:str):
    """
    Создание .gif файла
    """

    def animate(i):
        line.set_array(data_u[i])
        ax.set_title(f'Поле концентрации при t = {round(i*timestep,4)}')
        return line
    

    fig, ax = plt.subplots()
    line = plt.imshow(data_u[0], cmap = colour, extent = [0,L,0,H])
    plt.colorbar(line, ax=ax)                                                                   

    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_title('concentration field at t = {t}'.format(t=0))

    ani = animation.FuncAnimation(fig, animate, interval=timestep, blit=False, frames = data_u.shape[0])

    if showMe == True:
        plt.show()

    if picture == True:
        writer = animation.PillowWriter(
            fps=120, metadata=dict(artist='Приколисты'), bitrate=7200)
        ani.save(f"gifs/t={round(data_u.shape[0]*timestep,3)}, L={L}, H={H}.gif", writer=writer)


def generate_init(path:str, shapeX:int, shapeY:int, norm:float, reverse:bool):
    """
    Генерация НУ из картинки
    """

    image = Image.open(path).convert('L')
    if reverse==True:
        image = ImageOps.invert(image)
    new_image = image.resize((shapeX, shapeY))
    return np.asarray(new_image)/norm
