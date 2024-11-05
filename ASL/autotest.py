import os
import logging
import warnings

# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress warnings from Hugging Face Transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress all other warnings (including deprecation warnings)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from transformers import pipeline

# Load a pre-trained grammar correction model
corrector = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")

# Input sentence with spelling errors
text = "Hi my nameis yohan"

# Generate correction
result = result = corrector(text, max_new_tokens=10)


# Print the corrected text
print(result[0]['generated_text'])  # This should output: "Hi, my name is Yohan."
