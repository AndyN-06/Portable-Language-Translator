import google.generativeai as genai
import os

genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash-8b")

def correct_sentence(sentence):
    response = model.generate_content(f"Here are some common names to be aware of: Yohan, CJ, Andy, Cristian, Ryan. The only thing that you should response with is the corrected sentence, that is it. If needed, fix any grammar or spelling mistakes. If the sentence is already fully correct, just return the same sentence. Also try to add punctuation marks. Here is the sentence: {sentence}")
    return response.text

print(correct_sentence("That soumdss so exciting"))