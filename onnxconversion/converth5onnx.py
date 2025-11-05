import tensorflow as tf
import tf2onnx
import onnx

# Load the Keras model
keras_model = tf.keras.models.load_model(f'ckpt_200_50sec.h5')

# Convert the Keras model to ONNX format
onnx_model, _ = tf2onnx.convert.from_keras(keras_model)

# Save the ONNX model to a file
onnx.save_model(onnx_model, f'f16dubins_50sec.onnx')