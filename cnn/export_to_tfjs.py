import tensorflow as tf
import tensorflowjs as tfjs

# 1) Cargar el modelo completo (.h5) — debe contener arquitectura y pesos
model = tf.keras.models.load_model("hand_signs_cnn.h5")

# 2) Exportar a TensorFlow.js Layers format
#    Esto generará un directorio `public/model/` con model.json y los .bin
tfjs.converters.save_keras_model(model, "public/model")
print("Modelo exportado correctamente a public/model/")