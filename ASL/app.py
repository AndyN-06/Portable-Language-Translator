from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# Load tokenizer and model at startup
tokenizer = AutoTokenizer.from_pretrained("oliverguhr/spelling-correction-english-base")
model = AutoModelForSeq2SeqLM.from_pretrained("oliverguhr/spelling-correction-english-base")

class TextInput(BaseModel):
    text: str

@app.post("/correct-spelling/")
def correct_spelling(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"input_text": input.text, "output_text": output_text}
