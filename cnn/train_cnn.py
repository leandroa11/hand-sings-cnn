import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras import layers, models, callbacks

# ----------------------------
#  Parámetros de entrenamiento
# ----------------------------
IMAGE_SIZE = (224, 224)  # Tamaño de las imágenes de entrada (ancho, alto).
BATCH_SIZE = 32          # Tamaño del lote para el entrenamiento.
EPOCHS = 1               # Número de épocas para entrenar el modelo.

# Rutas de los datos
TRAIN_DIR = "data/train"        # Directorio con las imágenes de entrenamiento.
VAL_DIR = "data/validation"     # Directorio con las imágenes de validación.

# Nombre de los archivos de salida
keras_model = "hand_signs_cnn.h5"  # Archivo donde se guardará el modelo en formato H5.
tflite_model = "hand_signs.tflite" # Archivo donde se guardará el modelo en formato TFLite.

# ----------------------------
#  Preparación de los datos
# ----------------------------
# Generador de datos para entrenamiento con aumentación de datos.
train_datagen = ImageDataGenerator(
    rescale=1/255.,             # Normaliza los valores de los píxeles entre 0 y 1.
    rotation_range=15,          # Rotación aleatoria de hasta 15 grados.
    width_shift_range=0.1,      # Desplazamiento horizontal aleatorio de hasta el 10%.
    height_shift_range=0.1,     # Desplazamiento vertical aleatorio de hasta el 10%.
    zoom_range=0.1              # Zoom aleatorio de hasta el 10%.
)

# Generador de datos para validación (sin aumentación, solo normalización).
val_datagen = ImageDataGenerator(rescale=1/255.)

# Carga las imágenes de entrenamiento desde el directorio especificado.
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,     # Redimensiona las imágenes al tamaño especificado.
    batch_size=BATCH_SIZE,      # Tamaño del lote.
    class_mode="categorical"    # Clasificación multiclase.
)

# Carga las imágenes de validación desde el directorio especificado.
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,     # Redimensiona las imágenes al tamaño especificado.
    batch_size=BATCH_SIZE,      # Tamaño del lote.
    class_mode="categorical"    # Clasificación multiclase.
)

# Número de clases (categorías) detectadas en los datos de entrenamiento.
num_classes = train_gen.num_classes

# ----------------------------
#  Construcción del modelo
# ----------------------------
# Se utiliza MobileNetV2 como modelo base preentrenado en ImageNet.
base_model = MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),  # Tamaño de entrada de las imágenes (224x224x3).
    include_top=False,            # Excluye las capas superiores para personalizar la salida.
    weights="imagenet"            # Usa pesos preentrenados en el conjunto de datos ImageNet.
)

# Congelamos las capas del modelo base para evitar que se actualicen durante el entrenamiento inicial.
base_model.trainable = False

# Construcción del modelo completo utilizando capas adicionales.
model = models.Sequential([
    base_model,                          # Modelo base preentrenado.
    layers.GlobalAveragePooling2D(),    # Reduce las dimensiones espaciales, manteniendo la información más relevante.
    layers.Dropout(0.3),                # Dropout para reducir el sobreajuste.
    layers.Dense(64, activation="relu"), # Capa densa con 64 neuronas y activación ReLU.
    layers.Dense(num_classes, activation="softmax")  # Capa de salida con activación softmax para clasificación multiclase.
])

# Compilación del modelo:
# - Optimizer: Adam, que ajusta dinámicamente la tasa de aprendizaje.
# - Loss: Categorical Crossentropy, adecuada para problemas de clasificación multiclase.
# - Metrics: Accuracy, para evaluar el rendimiento del modelo.
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Muestra un resumen del modelo, incluyendo las capas y el número de parámetros entrenables.
model.summary()

# ----------------------------
#  Callbacks
# ----------------------------
# Lista de callbacks para el entrenamiento.
cb = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),  # Detiene el entrenamiento si no hay mejora.
    callbacks.ModelCheckpoint(keras_model, save_best_only=True, save_weights_only=False)  # Guarda el mejor modelo.
]

# ----------------------------
#  Entrenamiento
# ----------------------------
# Entrena el modelo utilizando los generadores de datos.
history = model.fit(
    train_gen,               # Datos de entrenamiento.
    validation_data=val_gen, # Datos de validación.
    epochs=EPOCHS,           # Número de épocas.
    callbacks=cb             # Callbacks para monitorear el entrenamiento.
)

# ---------------------------------------
#  Guardar modelo en formato TF 2.x standard
# ---------------------------------------
print(f"\nGuardando modelo Keras en '{keras_model}'...")

try:
    # Guarda el modelo en formato H5 utilizando el API de tf.keras.
    tf.keras.models.save_model(
        model,
        keras_model,
        overwrite=True,
        include_optimizer=True,
        save_format='h5'
    )
    print(f"Modelo guardado exitosamente en {keras_model}")
    
    # Verifica la estructura del modelo guardado.
    import time
    time.sleep(1)
    print("\nVerificando la estructura del modelo guardado...")
    import h5py
    try:
        with h5py.File(keras_model, 'r') as f:
            keys = list(f.keys())
            print("Contenido:", keys)
            
            if 'model_config' not in keys:
                print("ADVERTENCIA: model_config no encontrado. El modelo puede no ser compatible con TensorFlow.js")
            else:
                print("Verificación exitosa: model_config encontrado")
    except Exception as e:
        print(f"Error al abrir el archivo para verificación: {e}")

except Exception as e:
    print(f"Error al guardar el modelo con tf.keras.models.save_model: {e}")
    print("Intentando método alternativo de guardado...")
    
    try:
        # Alternativa: usar el método save() del modelo directamente.
        model.save(keras_model, save_format='h5', include_optimizer=True)
        print(f"Modelo guardado exitosamente con método alternativo en {keras_model}")
    except Exception as e2:
        print(f"Error en el método alternativo de guardado: {e2}")

# Si todas las opciones anteriores fallan, intentar guardarlo en formato SavedModel.
savedmodel_dir = "saved_model_dir"
try:
    print(f"\nGuardando modelo en formato SavedModel en '{savedmodel_dir}'...")
    tf.saved_model.save(model, savedmodel_dir)
    print(f"Modelo guardado exitosamente en formato SavedModel en {savedmodel_dir}")
except Exception as e:
    print(f"Error al guardar en formato SavedModel: {e}")

print("Entrenamiento y guardado completados.")