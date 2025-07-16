# convert_to_tflite.py

import tensorflow as tf

model = tf.keras.models.load_model("recycle_model.h5")

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open("recycle_model.tflite", "wb") as f:
    f.write(tflite_model)
