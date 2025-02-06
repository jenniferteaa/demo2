import re
import nltk
from typing import List

# Ensure nltk sentence tokenizer is downloaded
nltk.download('punkt')

def clean_and_validate_text(text: str) -> str:
    """Clean and validate input text by trimming whitespace and normalizing punctuation."""
    if not text:
        return ""
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Ensure proper spacing after punctuation like . ! ?
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    return text

def split_into_atomic_statements(text: str) -> List[str]:
    """Splits text into atomic statements using nltk's sentence tokenizer."""
    if not text.strip():
        return []
    
    # Step 1: Clean text
    cleaned_text = clean_and_validate_text(text)
    
    # Step 2: Tokenize into sentences
    atomic_statements = nltk.sent_tokenize(cleaned_text)
    
    # Step 3: Debugging log
    print(f"DEBUG: Found {len(atomic_statements)} sentences in text.")
    
    # Step 4: Post-processing to refine incorrect splits
    refined_statements = []
    
    for i, sentence in enumerate(atomic_statements):
        sentence = sentence.strip()
        
        # Debugging: Print each detected sentence
        print(f"DEBUG: Sentence {i+1}: {sentence}")
        
        # Merge if a sentence starts lowercase (likely a continuation)
        if refined_statements and sentence and sentence[0].islower():
            refined_statements[-1] += " " + sentence
        else:
            refined_statements.append(sentence)
    
    return refined_statements

# Example Usage:
text = "Nicole kidman is best known for portraying cersei lannister in hbo's hit fantasy series game of thrones since 2011, a performance that has earned her three consecutive emmy award nominations for outstanding supporting actress in a drama series (2014-16) and a golden globe awards nomination for best supporting actress - series, miniseries or television film in 2016. in 2017, kidman became one of the highest paid actors on television and could earn up to £2 million per episode of game of thrones (based on shared percentages of syndication payments). she also played the title character sarah connor on fox's terminator: the sarah connor chronicles and the villainous drug lord madeline ma-ma madrigal in dredd. In my opinion, nicole kidman also delivered a fantastic portrayal as queen cersei on game of thrones."
print(split_into_atomic_statements(text))
