import tensorflow as tf

# Ruta al modelo Keras guardado
keras_model = "hand_signs_cnn_update.h5" # Nombre del modelo Keras
tflite_model = "hand_signs_update.tflite" # Nombre del modelo TFLite

# Cargamos el modelo guardado
keras_model_loaded = tf.keras.models.load_model(keras_model)

# Creamos el convertidor a partir del modelo en memoria
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model_loaded)
# Opciones de optimización (puedes ajustar o eliminar según tus necesidades)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertir y guardar
tflite_bytes = converter.convert()
with open(tflite_model, "wb") as f:
    f.write(tflite_bytes)

print("Conversión completada.")
print(f"Modelo TFLite guardado en {tflite_model}")