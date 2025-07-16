# test_tflite_model.py

import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="recycle_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load image
img = Image.open("sample_data/test1.jpg").resize((160, 160))
img = np.array(img, dtype=np.float32) / 255.0
img = np.expand_dims(img, axis=0)

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
print("Prediction (0=Non-Recyclable, 1=Recyclable):", output)
