import numpy as np

def agregar_imagen(fondo, imagen, x, y):
    # verificar si la imagen tiene informacion de opacidad
    alto = imagen.shape[0]
    ancho = imagen.shape[1]
    if imagen.shape[-1] == 4:
        # normalizar la opacidad
        opacidad = imagen[:,:,3]/255
        # alpha blending
        
        # generar una imagen vacia
        imagen_3_canales = np.zeros((imagen.shape[0], imagen.shape[1], 3))
        
        # a cada canal multiplicarle la opacidad
        imagen_3_canales[:,:,0] = imagen[:,:,0] * opacidad
        imagen_3_canales[:,:,1] = imagen[:,:,1] * opacidad
        imagen_3_canales[:,:,2] = imagen[:,:,2] * opacidad

        # a la imagen de fondo, se le suma la imagen con informacion de opacidad
        fondo[y:y+alto, x:x+ancho, :] = (1-np.stack([opacidad, opacidad, opacidad], axis=-1)) * fondo[y:y+alto, x:x+ancho, :] + imagen_3_canales
    else:
        # reemplazamos la informacion del fondo
        fondo[y:y+alto, x:x+ancho, :] = imagen