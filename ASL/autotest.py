text = "HI NY NANE IS YOHA N"

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# 1. Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load the tokenizer and model with optimizations
tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large", use_fast=True)
model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")

# Move model to the appropriate device
model.to(device)

# If using GPU, convert model to half precision
if device.type == "cuda":
    model.half()

# Set model to evaluation mode and disable gradients
model.eval()
torch.set_grad_enabled(False)

# 3. Perform inference within no_grad context for efficiency
with torch.no_grad():
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generate edited text
    outputs = model.generate(
        inputs.input_ids,
        max_length=256,
        num_beams=1,             # Use greedy decoding for speed
        early_stopping=True      # Stop early when possible
    )
    
    # Decode the output tokens to text
    edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Edited Text:", edited_text)



