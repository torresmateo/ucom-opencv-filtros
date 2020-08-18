import cv2 as cv
import numpy as np
import os
import imageio
from utils import agregar_imagen

class Filtro(object):
    
    def __init__(self,
                 imagen):
        self.imagen = imageio.imread(imagen)
        self.cantidad_canales = (self.imagen.shape[-1]
                                 if len(self.imagen.shape) > 2 else 2)
        if self.cantidad_canales > 2:
            mascara2d = self.imagen.sum(axis=-1)
        else:
            mascara2d = self.imagen.copy()
        suma_c = mascara2d.sum(axis=0)
        suma_f = mascara2d.sum(axis=1)
        pos_i_x = (suma_c != 0).tolist().index(True)
        pos_f_x = (suma_c.shape[0] - (suma_c != 0)[::-1].tolist().index(True))
        pos_i_y = (suma_f != 0).tolist().index(True)
        pos_f_y = (suma_f.shape[0] - (suma_f != 0)[::-1].tolist().index(True))
        if self.cantidad_canales > 2:
            self.imagen = self.imagen[pos_i_y:pos_f_y, pos_i_x:pos_f_x, :]
        else:
            self.imagen = self.imagen[pos_i_y:pos_f_y, pos_i_x:pos_f_x]
        self.h, self.w = self.imagen.shape[:2]
        self.aspect_ratio = self.w / self.h

    def agregar_a_imagen(self, img, x, y, w, h):
        
        # ajustar el alto
        w_h = int(self.aspect_ratio * h)
        a_h = w_h * h
        # ajustar el ancho
        h_w = int(w / self.aspect_ratio)
        a_w = w * h_w 
        # tomar el ajuste que genere la mayor area
        if a_h > a_w:
            rw, rh = (w_h, h)
            rx, ry = x - np.abs(int((w - rw)/2)), y
        else:
            rw, rh = (w, h_w)
            rx, ry = x, y - np.abs(int((h - rh)/2))
        imagen_chica = cv.resize(self.imagen, (rw, rh))
        agregar_imagen(img, imagen_chica, rx, ry)