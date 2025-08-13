import os

# Set the Keras backend to TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
print(f"Keras backend: {keras.backend.backend()}")

# You can now import and use Keras layers and models
# from either `keras` or `tensorflow.keras`