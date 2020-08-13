import numpy as np
import cv2 as cv

def agregar_imagen(fondo, imagen, x, y):
    
    # TODO: verificar el caso en que la imagen empieza (o termina) fuera de el 
    # rango del fondo
    
    # verificar si la imagen tiene informacion de opacidad
    alto = imagen.shape[0]
    ancho = imagen.shape[1]
    cantidad_canales = imagen.shape[-1]
    if cantidad_canales == 4:
        # normalizar la opacidad
        opacidad = imagen[:,:,3]/255
        # alpha blending
        
        # generar una imagen vacia
        imagen_3_canales = np.zeros((imagen.shape[0], imagen.shape[1], 3))
        
        # a cada canal multiplicarle la opacidad
        imagen_3_canales[:,:,0] = imagen[:,:,0] * opacidad
        imagen_3_canales[:,:,1] = imagen[:,:,1] * opacidad
        imagen_3_canales[:,:,2] = imagen[:,:,2] * opacidad

        # a la imagen de fondo, se le suma 
        # la imagen con informacion de opacidad
        opacidad_3_canales = np.stack([opacidad, opacidad, opacidad],
                                      axis=-1)
        fondo[y:y+alto, x:x+ancho, :] = ((1 - opacidad_3_canales) * 
                                         fondo[y:y+alto, x:x+ancho, :] + 
                                         imagen_3_canales)
    elif cantidad_canales == 3:
        # reemplazamos la informacion del fondo
        fondo[y:y+alto, x:x+ancho, :] = imagen
    else: # asumir que la cantidad de canales es 1
        imagen_3_canales = cv.cvtColor(imagen, cv.COLOR_GRAY2RGB)
        fondo[y:y+alto, x:x+ancho, :] = imagen_3_canales