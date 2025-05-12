import os
import cv2
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras import layers, models, callbacks

# ----------------------------
#  Parámetros de entrenamiento
# ----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 1

# Rutas de los datos
TRAIN_DIR = "data/train"
VAL_DIR   = "data/validation"

# Nombre de los archivos de salida
keras_model  = "hand_signs_cnn.h5"
tflite_model = "hand_signs.tflite"

# ----------------------------
#  Preparación de los datos
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
val_datagen = ImageDataGenerator(rescale=1/255.)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = train_gen.num_classes

# ----------------------------
#  Construcción del modelo
# ----------------------------
base_model = MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # congelamos la base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
#  Callbacks
# ----------------------------
cb = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint(keras_model, save_best_only=True, save_weights_only=False)
]

# ----------------------------
#  Entrenamiento
# ----------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=cb
)

# ---------------------------------------
#  Guardar modelo en formato TF 2.x standard
# ---------------------------------------
print(f"\nGuardando modelo Keras en '{keras_model}'...")

try:
    # Asegurar que usamos el API de tf.keras directamente
    # Simplificamos los parámetros para mayor compatibilidad
    tf.keras.models.save_model(
        model,
        keras_model,
        overwrite=True,
        include_optimizer=True,
        save_format='h5'
    )
    print(f"Modelo guardado exitosamente en {keras_model}")
    
    # Dar tiempo para que el sistema de archivos complete la operación
    import time
    time.sleep(1)
    
    # Verificar que el modelo tenga la estructura correcta
    print("\nVerificando la estructura del modelo guardado...")
    import h5py
    try:
        with h5py.File(keras_model, 'r') as f:
            keys = list(f.keys())
            print("Contenido:", keys)
            
            if 'model_config' not in keys:
                print("ADVERTENCIA: model_config no encontrado. El modelo puede no ser compatible con TensorFlow.js")
                print("Más información sobre el contenido del archivo:")
                for key in keys:
                    print(f"- {key} y sus elementos: {list(f[key].keys()) if hasattr(f[key], 'keys') else 'No tiene subelementos'}")
            else:
                print("Verificación exitosa: model_config encontrado")
    except Exception as e:
        print(f"Error al abrir el archivo para verificación: {e}")
        print("El modelo se guardó pero no se pudo verificar su estructura")

except Exception as e:
    print(f"Error al guardar el modelo con tf.keras.models.save_model: {e}")
    print("Intentando método alternativo de guardado...")
    
    try:
        # Alternativa: usar el método save() del modelo directamente
        model.save(keras_model, save_format='h5', include_optimizer=True)
        print(f"Modelo guardado exitosamente con método alternativo en {keras_model}")
        
        # Dar tiempo para que el sistema de archivos complete la operación
        import time
        time.sleep(1)
    except Exception as e2:
        print(f"Error en el método alternativo de guardado: {e2}")

# Si todas las opciones anteriores fallan, intentar guardarlo en formato SavedModel
# que también es compatible con TensorFlow.js
savedmodel_dir = "saved_model_dir"
try:
    print(f"\nGuardando modelo en formato SavedModel en '{savedmodel_dir}'...")
    tf.saved_model.save(model, savedmodel_dir)
    print(f"Modelo guardado exitosamente en formato SavedModel en {savedmodel_dir}")
    print("Este formato también es compatible con TensorFlow.js")
    print("Para convertirlo, usa: tensorflowjs_converter --input_format=tf_saved_model saved_model_dir tfjs_model_dir")
except Exception as e:
    print(f"Error al guardar en formato SavedModel: {e}")

print("Entrenamiento y guardado completados.")