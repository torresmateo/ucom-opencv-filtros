import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# elegir el directorio donde se encuentran las imágenes de entrenamiento
train_dir = './entrenamiento'

# crear una instancia de ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# configurar el generador con el directorio correcto
train_gen = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(256,256),  # indicamos al generador que cmabie la resolución a 256x256 píxeles
    class_mode='sparse')  # indicamos que es un dataset para clasificación binaria

# crear el modelo
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(256,256,3)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(33, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation='relu'), 
                                    tf.keras.layers.Dense(4, activation='softmax')])

# compilar el modelo
# tu código aquí (~1 línea)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# entrenamos el modelo en 100 epochs - va a tardar unos minutos, aprovecha el tiempo para leer sobre generadores!
historia = model.fit(
      train_gen,
      epochs=5,
      verbose=1)

model.save('./modelos/detector_gestos.h5')