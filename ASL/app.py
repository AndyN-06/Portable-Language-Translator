# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import os
import logging
import warnings

app = FastAPI(title="Grammar Correction API", version="1.0")

# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress warnings from Hugging Face Transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress all other warnings (including deprecation warnings)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load the model once at startup
corrector = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")

class TextInput(BaseModel):
    text: str

class CorrectionOutput(BaseModel):
    corrected_text: str

@app.post("/correct", response_model=CorrectionOutput)
def correct_text(input: TextInput):
    """
    Corrects the grammar and spelling of the input text.
    """
    try:
        result = corrector(input.text, max_new_tokens=10)
        corrected_text = result[0]['generated_text']
        return CorrectionOutput(corrected_text=corrected_text)
    except Exception as e:
        # Handle exceptions gracefully
        return CorrectionOutput(corrected_text=f"Error: {str(e)}")
