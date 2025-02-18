import tensorflow as tf

# 1) Load the Keras model
model = tf.keras.models.load_model('my_model.keras')

# 2) Create a TFLiteConverter from the loaded model
converter = tf.lite.TFLiteConverter.from_saved_model(model)

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.allow_custom_ops = False
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Set input shape explicitly if known
converter.target_spec.supported_types = [tf.float32]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)


print("Conversion successful! Model saved to model.tflite.")
