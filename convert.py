import tensorflow as tf

# 1) Load the Keras model
model = tf.keras.models.load_model('my_model.keras')

# 2) Create a TFLiteConverter from the loaded model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3) Allow Select TF Ops, so TFLite can handle ops that do not have native TFLite equivalents
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# 4) Disable the lower_tensor_list_ops so it won't try to convert certain LSTM internals
converter._experimental_lower_tensor_list_ops = False

# 5) Enable resource variables
converter.experimental_enable_resource_variables = True

# 6) Convert
tflite_model = converter.convert()

# 7) Save the .tflite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Conversion successful! Model saved to model.tflite.")
