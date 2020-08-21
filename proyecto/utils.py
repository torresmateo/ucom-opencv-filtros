import numpy as np
import cv2 as cv

def agregar_imagen(fondo, imagen, x, y):
    
    filtro_x = -x if x < 0 else 0
    filtro_y = -y if y < 0 else 0
    
    x = 0 if x < 0 else x
    y = 0 if y < 0 else y
    
    if len(imagen.shape) > 2:
        alto, ancho, cantidad_canales = imagen.shape
    else:
        alto, ancho = imagen.shape
        cantidad_canales = 2

    fondo_alto, fondo_ancho, _ = fondo.shape
    
    filtro_ancho = fondo_ancho - x if x + ancho > fondo_ancho else ancho
    filtro_alto = fondo_alto - y if y + alto > fondo_alto else alto
    
    if cantidad_canales > 2:
        imagen_cortada = imagen[filtro_y:filtro_y+filtro_alto,
                                filtro_x:filtro_x+filtro_ancho, :]
    else:
        imagen_cortada = imagen[filtro_y:filtro_y+filtro_alto,
                                filtro_x:filtro_x+filtro_ancho]
        
    alto, ancho = imagen_cortada.shape[0], imagen_cortada.shape[1]
        
    # verificar si la imagen tiene informacion de opacidad
    if cantidad_canales == 4:
        # normalizar la opacidad
        opacidad = imagen_cortada[:,:,3]/255
        # alpha blending
        
        # generar una imagen vacia
        imagen_3_canales = np.zeros((imagen_cortada.shape[0], 
                                     imagen_cortada.shape[1], 3))
        
        # a cada canal multiplicarle la opacidad
        imagen_3_canales[:,:,0] = imagen_cortada[:,:,0] * opacidad
        imagen_3_canales[:,:,1] = imagen_cortada[:,:,1] * opacidad
        imagen_3_canales[:,:,2] = imagen_cortada[:,:,2] * opacidad

        # a la imagen de fondo, se le suma 
        # la imagen con informacion de opacidad
        opacidad_3_canales = np.stack([opacidad, opacidad, opacidad],
                                      axis=-1)
        fondo[y:y+alto, x:x+ancho, :] = ((1 - opacidad_3_canales) * 
                                         fondo[y:y+alto, x:x+ancho, :] + 
                                         imagen_3_canales)
    elif cantidad_canales == 3:
        # reemplazamos la informacion del fondo
        fondo[y:y+alto, x:x+ancho, :] = imagen_cortada
    else: # asumir que la cantidad de canales es 1
        imagen_3_canales = cv.cvtColor(imagen_cortada, cv.COLOR_GRAY2RGB)
        fondo[y:y+alto, x:x+ancho, :] = imagen_3_canales


def calcular_margenes(x,y,w,h,MARGEN_X, MARGEN_Y):
    pad_w = int(w * MARGEN_X)
    pad_h = int(h * MARGEN_Y)

    x_pad = x - pad_w
    y_pad = y - pad_h
    w_pad = w + 2*pad_w
    h_pad = h + 2*pad_h

    return (x_pad, y_pad, w_pad, h_pad)

# sacado de imutils
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH)), M