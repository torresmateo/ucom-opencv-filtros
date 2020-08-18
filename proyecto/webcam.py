import cv2 as cv
import imageio
import os
from utils import agregar_imagen
from filtro import Filtro

script_dir = os.path.dirname(os.path.realpath(__file__))
image_dir = os.path.join(script_dir, 'imagenes')

mascara = os.path.join(image_dir,"stormtrooper-ojos-transparentes.png")
mascara2 = os.path.join(image_dir,"stormtrooper.png")

filtros = [Filtro(mascara), 
           Filtro(mascara2),
           Filtro(mascara2)]

haar_cascade_file = os.path.join(script_dir, 
                                 'modelos', 
                                 'haarcascade_frontalface_default.xml')
detector_caras = cv.CascadeClassifier(haar_cascade_file)

# capturar imagen desde la webcam
cap = cv.VideoCapture(0)

# PARAMETROS PARA AJUSTAR EL BORDE DE LA CARA AL BORDE DE LA CABEZA
AGREGAR_X_PORCENTAJE = 0.05
AGREGAR_Y_PORCENTAJE = 0.15

while True:
    ret, img = cap.read() # leer la webcam
    img = cv.flip( img, 1 ) # flip horizontal para que sea un espejo
    
    # convertir el frame a escala de grises
    img_gris = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    
    # detectar las caras
    caras = detector_caras.detectMultiScale(img_gris, 1.3, 5)
    
    # ordenar las caras con respecto al eje X, de tal manera a mantener un 
    # "estado" que se pueda asociar a cada cara
    # if len(caras) > 1: # se detectaron múltiples caras
    #    caras = caras[caras[0,:].argsort()]
    

    
    for i, (x, y, w, h) in enumerate(caras):
        
        pad_w = int(w * AGREGAR_X_PORCENTAJE)
        pad_h = int(h * AGREGAR_Y_PORCENTAJE)
        
        x_pad = x - pad_w
        y_pad = y - pad_h
        w_pad = w + 2*pad_w
        h_pad = h + 2*pad_h

        filtros[i].agregar_a_imagen(img, x_pad, y_pad, w_pad, h_pad)
        cv.rectangle(img,(x,y), (x+w, y+h), (255,0,0)) # la cara detectada

        # el rectangulo con margen
        cv.rectangle(img,(x_pad,y_pad), (x_pad+w_pad, y_pad+h_pad), (0,0,255))
    cv.putText(img,
               f'AGREGAR_X_PORCENTAJE: {AGREGAR_X_PORCENTAJE}',
               (20,30), 
               cv.FONT_HERSHEY_COMPLEX,
               0.71, 
               (0,0,255),
               2)
    cv.putText(img,
               f'AGREGAR_Y_PORCENTAJE: {AGREGAR_Y_PORCENTAJE}',
               (20,50), 
               cv.FONT_HERSHEY_COMPLEX,
               0.71, 
               (0,0,255),
               2)
    
    cv.imshow('Ttulo de la ventana', img)

    k = cv.waitKey(30)
    if k == 27: # ESC (ASCII)
        break
    elif k == ord('y'):
        AGREGAR_Y_PORCENTAJE += 0.01
    elif k == ord('u'):
        AGREGAR_Y_PORCENTAJE -= 0.01
cap.release()
cv.destroyAllWindows()