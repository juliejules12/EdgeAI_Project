# Edge AI Prototype - Recyclable Item Classification

This project demonstrates Edge AI using a lightweight image classifier built with TensorFlow and converted to TensorFlow Lite for deployment on edge devices like Raspberry Pi.

## Project Structure

- `model_training.ipynb`: Train a MobileNetV2-based model.
- `convert_to_tflite.py`: Convert to .tflite format.
- `test_tflite_model.py`: Run local inference on a test image.

## Dataset

Use any binary image classification dataset (e.g., recyclable vs. non-recyclable). Organize as:
