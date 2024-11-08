from spellchecker import SpellChecker

def correct_text(text):
    spell = SpellChecker()
    
    # Split into words
    words = text.split()
    
    # Find misspelled words
    misspelled = spell.unknown(words)
    
    # Replace misspelled words with corrections
    corrected_words = []
    for word in words:
        if word in misspelled:
            corrected_words.append(spell.correction(word))
        else:
            corrected_words.append(word)
    
    # Join back into text
    return ' '.join(corrected_words)

# Test the function
text = "HI MY NMAME ISS"
text = text.lower()
corrected = correct_text(text)
print(f"Original: {text}")
print(f"Corrected: {corrected}")
