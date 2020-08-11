import cv2 as cv
import imageio
import os
from utils import agregar_imagen

script_dir = os.path.dirname(os.path.realpath(__file__))
image_dir = os.path.join(script_dir, 'imagenes')

mascara = imageio.imread(os.path.join(image_dir,"stormtrooper-ojos-transparentes.png"))

# capturar imagen desde la webcam
cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read() # leer la webcam
    img = cv.flip( img, 1 ) # flip horizontal para que sea un espejo
    
    agregar_imagen(img, mascara, 200, 200)

    cv.imshow('Ttulo de la ventana', img)

    k = cv.waitKey(30)
    if k == 27: # ESC (ASCII)
        break
cap.release()
cv.destroyAllWindows()