import cv2 as cv
import numpy as np
import os
import imageio
from utils import agregar_imagen

class Filtro(object):
    
    def __init__(self,
                 imagen, 
                 offset_ancho, 
                 offset_x,
                 offset_y):
        self.imagen = imageio.imread(imagen)
        self.offset_ancho = offset_ancho
        self.offset_x = offset_x
        self.offset_y = offset_y
        
    def agregar_a_imagen(self, img, x, y, w, h):
        
        # TODO: mejorar la geometria del escalado
        factor_de_resize = (w+self.offset_ancho)/self.imagen.shape[0]
        
        # coordenada del punto a alinear en el nuevo tamaño
        offset_x_resize = int(factor_de_resize * self.offset_x)
        offset_y_resize = int(factor_de_resize * self.offset_y)
        
        imagen_chica = cv.resize(self.imagen, 
                                 (w + self.offset_ancho, 
                                  w + self.offset_ancho))
        agregar_imagen(img, imagen_chica, 
                       x - offset_x_resize, 
                       y - offset_y_resize)