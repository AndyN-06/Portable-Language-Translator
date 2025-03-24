import tensorflow as tf

model = tf.keras.models.load_model("hands3.keras")

# Export as SavedModel
model.export("newest")

# Then build a converter from SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model("newest")
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

with open("newest.tflite", "wb") as f:
    f.write(tflite_model)
