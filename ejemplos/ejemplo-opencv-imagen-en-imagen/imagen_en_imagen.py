import cv2 as cv
import imageio
import os


script_dir = os.path.dirname(os.path.realpath(__file__))
image_dir = os.path.join(script_dir, 'imagenes-para-mostrar')
# cargar las imagenes en memoria
# jpg (no tiene información de transparencia)
cerveza = imageio.imread(os.path.join(image_dir,"cerveza.jpg"))

# png (tiene información de tranparencia)
arroz = imageio.imread(os.path.join(image_dir,"rice-bowl.png"))

pc = {
    'x':20, 'y':20,
    'w':200, 'h':200
} 
pa = {
    'x': 320, 'y': 20,
    'w': 200, 'h': 200
}

cerveza_chica = cv.resize(cerveza, (pc['w'], pc['h']) ) 
arroz_chico = cv.resize(arroz, (pa['w'], pa['h']) ) 

# capturar imagen desde la webcam
cap = cv.VideoCapture(0)

def agregar_imagen_a_imagen(img_a, img_b, x, y):
    # determinar si la imagen b tiene información de la opacidad
    if img_b.shape[-1] == 4: # existe canal de opacidad
        
        # pixeles que no son transparentes
        mask = (img_b != 0).all(axis=2)
        
        # la porcion de la imagen A que reemplazaria con o sin transparencia
        bg = img_a[y:y+img_b.shape[0], x:x+img_b.shape[1], :]
        
        # copiando pixeles no transparentes de B a A
        bg[mask] = img_b[mask, :3]  
        
        img_a[y:y+img_b.shape[0], x:x+img_b.shape[1], :] = bg
    else:
        img_a[y:y+img_b.shape[0], x:x+img_b.shape[1], :] = img_b

while True:
    ret, img = cap.read() # leer la webcam
    img = cv.flip( img, 1 ) # flip horizontal para que sea un espejo

    # obtener una version chica de las imagenes

    
    # img[
    #     pc['y']:pc['y'] + pc['h']+1, # y : y + h (el rango en el eje y que se va a reemplazar)
    #     pc['x']:pc['x'] + pc['w'], # x : x + h (el rango en el eje y que se va a reemplazar)
    #     :
    # ] = cerveza_chica
    
    #agregar_imagen_a_imagen(img, cerveza_chica, pc['x'], pc['y'])

    agregar_imagen_a_imagen(img, arroz_chico, pa['x'], pa['y'])


    cv.imshow('Ttulo de la ventana',img)

    k = cv.waitKey(30)
    if k == 27: # ESC (ASCII)
        break
cap.release()
cv.destroyAllWindows()

