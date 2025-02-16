import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("my_model.keras")

# Convert to SavedModel format
model.save("my_model_saved", save_format="tf")
