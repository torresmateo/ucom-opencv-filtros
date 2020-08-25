import cv2 as cv
import os
from utils import agregar_imagen, calcular_margenes
from filtro import Filtro
import tensorflow as tf
import numpy as np


script_dir = os.path.dirname(os.path.realpath(__file__))
image_dir = os.path.join(script_dir, 'imagenes')

mascara = os.path.join(image_dir,"stormtrooper-ojos-transparentes.png")
mascara2 = os.path.join(image_dir,"stormtrooper-2.png")

filtros = [Filtro(mascara, np.array([325, 224]), np.array([179, 225])), 
           Filtro(mascara2, np.array([325, 224]), np.array([179, 225])),
           Filtro(mascara2, np.array([325, 224]), np.array([179, 225])),
           Filtro(mascara2, np.array([325, 224]), np.array([179, 225])),
           ]

haar_cascade_file = os.path.join(script_dir, 
                                 'modelos', 
                                 'haarcascade_frontalface_default.xml')
detector_caras = cv.CascadeClassifier(haar_cascade_file)
detector_keypoints = tf.keras.models.load_model(
    os.path.join(script_dir, 'modelos', 'pre-trained.h5'))
detector_gestos = tf.keras.models.load_model(
    os.path.join(script_dir, 'modelos', 'detector_gestos.h5'))

# capturar imagen desde la webcam
cap = cv.VideoCapture(0)

# PARAMETROS PARA AJUSTAR EL BORDE DE LA CARA AL BORDE DE LA CABEZA
AGREGAR_X_PORCENTAJE = 0.05
AGREGAR_Y_PORCENTAJE = 0.15

KEYPOINT_X_PORCENTAJE = 0.05
KEYPOINT_Y_PORCENTAJE = 0.05

FILTRO_ACTIVADO = False
FILTRO_ACTUAL = 0
PRINT_DEBUG = False

FRAME = 0

CLASES_GESTOS = {
    0:'fondo',
    1:'gestoA',
    2:'gestoB',
    3:'gestoC'
}

while True:
    ret, img = cap.read() # leer la webcam
    img = cv.flip( img, 1 ) # flip horizontal para que sea un espejo
    
    if FILTRO_ACTIVADO:
        # convertir el frame a escala de grises
        img_gris = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        
        # detectar las caras
        caras = detector_caras.detectMultiScale(img_gris, 1.3, 5)
        
        # ordenar las caras con respecto al eje X, de tal manera a mantener un 
        # "estado" que se pueda asociar a cada cara
        # if len(caras) > 1: # se detectaron múltiples caras
        #    caras = caras[caras[0,:].argsort()]
        
        for i, (x, y, w, h) in enumerate(caras):
            
            # DETECCION DE KEYPOINTS
            
            # extraer la cara
            x_cara, y_cara, w_cara, h_cara = calcular_margenes(
                x, y, w, h, KEYPOINT_X_PORCENTAJE, KEYPOINT_Y_PORCENTAJE)
            cara = img[y_cara:y_cara + h_cara, x_cara:x_cara+w_cara]
            # convertir a escala de grises
            cara = cv.cvtColor(cara, cv.COLOR_RGB2GRAY)
            # hacer un resize a 96, 96
            cara_chica = cv.resize(cara, (96, 96))
            # transformar el array a una dimensionalidad compatible con el modelo
            cara_input = np.array([cara_chica.reshape(96, 96, 1)]) / 255
            # predecir
            keypoints = detector_keypoints.predict(cara_input) * 96
            # traducir los keypoints a el tamaño y posición original
            
            factor_cara_x = w_cara / 96
            factor_cara_y = h_cara / 96
            if PRINT_DEBUG:
                agregar_imagen(img, cara_chica, 20, 150)
            keypoints = keypoints.astype(np.int32).reshape(15, 2)
            
            keypoints_original = keypoints.copy()
            keypoints_original[:, 0] = keypoints_original[:, 0] * factor_cara_x + x_cara
            keypoints_original[:, 1] = keypoints_original[:, 1] * factor_cara_y + y_cara
            keypoints_original = keypoints_original.astype(np.int32)
            if PRINT_DEBUG:
                for j in range(2): # lo equivalente a los centros de los ojos
                    c = tuple(keypoints[j] + np.array([20, 150]))
                    c_original = tuple(keypoints_original[j])
                    cv.circle(img, c, 2, (0,255,0))
                    cv.circle(img, c_original, 4, (0,0,255), -1)
            

            x_pad, y_pad, w_pad, h_pad = calcular_margenes(x, y, w, h, 
                                                        AGREGAR_X_PORCENTAJE, 
                                                        AGREGAR_Y_PORCENTAJE)

            # AGREGAR EL FILTRO A LA CARA
            #filtros[i].agregar_a_imagen(img, x_pad, y_pad, w_pad, h_pad)
            filtros[FILTRO_ACTUAL].agregar_a_imagen_kp(img, keypoints_original)
            
            cv.rectangle(img,(x,y), (x+w, y+h), (255,0,0)) # la cara detectada

            # el rectangulo con margen
            if PRINT_DEBUG:
                cv.rectangle(img,(x_pad,y_pad), (x_pad+w_pad, y_pad+h_pad), (0,0,255))
    if PRINT_DEBUG:
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
    elif k == ord('a'):
        FILTRO_ACTUAL = (FILTRO_ACTUAL + 1) % len(filtros)
    elif k == ord('f'):
        FILTRO_ACTIVADO = ~FILTRO_ACTIVADO
    
    # if FRAME % 10 == 0: # cada 10 frames
    #     img_h, img_w = img.shape[:2]
    #     x = img_w - 400
    #     y = 0
    #     w = 400
    #     h = 400
    #     gesto = img[y:y+h, x:x+w]
    #     gesto = cv.resize(gesto, (256, 256))
    #     gesto_RGB = cv.cvtColor(gesto, cv.COLOR_BGR2RGB)
    #     gesto_RGB = np.array([gesto_RGB])
    #     clase = np.argmax(detector_gestos.predict(gesto_RGB)[0])
    #     print(f'clase = {clase}')
    #     if CLASES_GESTOS[clase] == 'gestoA':
    #         FILTRO_ACTUAL = (FILTRO_ACTUAL + 1) % len(filtros)
    #     elif CLASES_GESTOS[clase] == 'gestoB':
    #         FILTRO_ACTIVADO = ~FILTRO_ACTIVADO
    #     elif CLASES_GESTOS[clase] == 'gestoC':
    #         PRINT_DEBUG = ~PRINT_DEBUG
        
cap.release()
cv.destroyAllWindows()