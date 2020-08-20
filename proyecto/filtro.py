import cv2 as cv
import numpy as np
import os
import imageio
from utils import agregar_imagen

class Filtro(object):
    
    def __init__(self,
                 imagen, 
                 ojo_izq, # np.array de 2 dimensiones
                 ojo_der):
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
        
        self.ojo_izq = ojo_izq
        self.ojo_der = ojo_der
        self.distancia_ojos = np.linalg.norm(self.ojo_izq - self.ojo_der)

    def agregar_a_imagen(self, img, x, y, w, h):
        """agregar el filtro a la imagen img con respecto a un bounding box
        definido por x, y, w, h
        """
        
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
    
    def agregar_a_imagen_kp(self, img, keypoints):
        """agregar el filtro a la imagen img con respecto a coordenadas del
        keypoint (CENTRO DE LOS OJOS)
        """
        ojo_izq = keypoints[0]
        ojo_der = keypoints[1]
        
        # distancia de los ojos en la imagen original
        # calcular la escala
        distancia_original = np.linalg.norm(ojo_izq - ojo_der)
        
        diff_distancias = distancia_original - self.distancia_ojos
        
        nueva_pos_izq = ojo_izq
        nueva_pos_der = self.ojo_der + diff_distancias

        # calcular el ángulo
        B = ojo_der - ojo_izq # para usar el ojo izq como origen
        B = B/np.linalg.norm(B) # normalizar
        B_p = self.ojo_der - self.ojo_izq
        B_p = B_p/np.linalg.norm(B_p) # normalizar
        angulo = np.arccos(np.dot(B, B_p) / np.linalg.norm(B) * np.linalg.norm(B_p))
        rot_mat = cv.getRotationMatrix2D(tuple(self.ojo_izq), angulo, 1)
        result = cv.warpAffine(self.imagen, rot_mat, self.imagen.shape[1::-1], flags=cv.INTER_LINEAR)
        agregar_imagen(img, result, 400, 200)
        # EXPLICACION
        # calcular el angulo y la escala entre dos puntos en el filtro 
        # y alinear a ojo_izq y ojo_der
        
        # alinear self.ojo_izq a ojo_izq y lo mismo para ojo_der y self.ojo_der
        # unicamente calculando un ANGULO y UNA ESCALA
        