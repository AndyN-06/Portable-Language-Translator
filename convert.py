import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("my_model.keras")

# Export the model to SavedModel format
model.export("my_model_saved")
