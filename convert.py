import tensorflow as tf

model = tf.keras.models.load_model("hands.keras")

# Export as SavedModel
model.export("hands_saved_model")

# Then build a converter from SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model("hands_saved_model")
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

with open("hands.tflite", "wb") as f:
    f.write(tflite_model)
