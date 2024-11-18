import google.generativeai as genai
import os

genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash-8b")

def correct_sentence(sentence):
    response = model.generate_content(f"Here are some common names to be aware of: Yohan, CJ, Andy, Cristian, Ryan. The only thing that you should response with is the corrected sentence, that is it. Fix any grammar or spelling mistakes: {sentence}")
    return response.text