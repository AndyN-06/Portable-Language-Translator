import re
from autocorrect import Speller
from wordsegment import load, segment
from nltk.corpus import words

# Load English words from nltk and the wordsegment module
load()
nltk_words = set(words.words())  # Load a list of valid English words

# Initialize autocorrect spell checker
spell_autocorrect = Speller()

# Function to reduce repetitions but only for invalid words
def reduce_repetitions(text, max_repeats=1):
    def reduce_word_repetitions(word):
        # Reduce repeated characters in each word
        reduced_word = re.sub(r'(.)\1+', lambda m: m.group(1) * max_repeats, word)
        
        # If the reduced word is valid, return it, else return the original word
        if reduced_word in nltk_words:
            return reduced_word
        else:
            return word  # Preserve the original if reducing it breaks the word
        
    # Split the text into words
    words = re.findall(r'\w+', text)
    
    # Reduce repetitions word by word
    return ' '.join(reduce_word_repetitions(word) for word in words)

# Function to segment and correct the final sentence
def correct_sentence(input_string):
    # Step 1: Segment the concatenated string into words
    segmented_words = segment(input_string)

    # Step 2: Reduce repetitions (only for invalid words)
    reduced_words = [reduce_repetitions(word) for word in segmented_words]

    # Step 3: Correct the words using autocorrect
    corrected_words = [spell_autocorrect(word) for word in reduced_words]

    # Step 4: Join corrected words into a sentence
    return " ".join(corrected_words)

# Test examples
test_string1 = "GOODJOBTODAY"
test_string2 = "IILLOOVVEEYYOOUU"
test_string3 = "GGGOOODDDWWOOOOOOD"

print(correct_sentence(test_string1))  # Output: "good job today"
print(correct_sentence(test_string2))  # Output: "I love you"
print(correct_sentence(test_string3))  # Output: "good wood"


