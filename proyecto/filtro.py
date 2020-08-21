import cv2 as cv
import numpy as np
import os
import imageio
from utils import agregar_imagen, rotate_bound

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
        
        self.ojo_izq = ojo_izq - np.array([pos_i_x, pos_i_y])
        self.ojo_der = ojo_der - np.array([pos_i_x, pos_i_y])
        self.pendiente = (np.abs(self.ojo_izq[1] - self.ojo_der[1]) /
                          np.abs(self.ojo_izq[0] - self.ojo_der[0]))
        self.angulo = np.arctan(self.pendiente)
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

        pendiente = (np.abs(ojo_izq[1] - ojo_der[1]) /
                     np.abs(ojo_izq[0] - ojo_der[0]))
        angulo_fondo = np.arctan(pendiente)
        delta_angulos = np.degrees(self.angulo - angulo_fondo)

        imagen = self.imagen.copy()
        imagen = cv.circle(imagen, tuple(self.ojo_izq), 4, (255,0,0), -1)
        imagen = cv.circle(imagen, tuple(self.ojo_der), 4, (0,255,0), -1)
        # rotacion
        img_rotada, M = rotate_bound(imagen, delta_angulos)
        ojo_ir = np.squeeze(cv.transform(self.ojo_izq.reshape(1,1,-1), M))
        ojo_dr = np.squeeze(cv.transform(self.ojo_der.reshape(1,1,-1), M))
        # escalamiento
        dis_fondo = np.linalg.norm(ojo_izq - ojo_der)
        ratio = dis_fondo / self.distancia_ojos
        img_escalada = cv.resize(img_rotada, None, fx=ratio, fy=ratio,
                                 interpolation=cv.INTER_CUBIC)
        # translación
        posicion_filtro = (ojo_izq - ojo_ir * ratio).astype(np.int32)
        agregar_imagen(img, img_escalada, 
                       posicion_filtro[0], posicion_filtro[1])
        # EXPLICACION
        # calcular el angulo y la escala entre dos puntos en el filtro 
        # y alinear a ojo_izq y ojo_der
        
        # alinear self.ojo_izq a ojo_izq y lo mismo para ojo_der y self.ojo_der
        # unicamente calculando un ANGULO y UNA ESCALA
        