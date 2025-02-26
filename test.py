import tensorflow as tf
from tensorflow.lite.experimental import load_delegate

try:
    gpu_delegate = load_delegate("libtensorflowlite_gpu_delegate.so")
    print("GPU delegate loaded successfully.")
except Exception as e:
    print("Failed to load GPU delegate:", e)