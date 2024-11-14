def load_model():
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer and model
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

    return tokenizer, model, device

def fix_text(input_text, tokenizer, model, device):
    import torch

    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate the corrected text
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=512)
    
    # Decode the output ids to text
    corrected_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return corrected_text