# We build a vocabulary by tokenizing the entire text in a training dataset into individual tokens. 
# These individual tokens are then sorted alphabetically, and duplicate tokens are removed. 
# The unique tokens are then aggregated into a vocabulary that defines a mapping from each unique token to a unique integer value. 
#The depicted vocabulary is purposefully small for illustration purposes and contains no punctuation or special characters for simplicity.

import re

# The encode function turns text into token IDs
# The decode function turns token IDs back into text
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


with open("the-verdict.txt", "r", encoding="utf-8") as f: raw_text = f.read()
print("Total number of character:", len(raw_text)) 
#print(raw_text[:99])

preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text) 
preprocessed = [item.strip() for item in preprocessed if item.strip()] 
print(len(preprocessed))

# Let's print the first 30 tokens for a quick visual check:
#print(preprocessed[:30])

all_words = sorted(list(set(preprocessed))) 
vocab_size = len(all_words) 
print(vocab_size)

# After determining that the vocabulary size is 1,159 via the above code, 
# we create the vocabulary and print its first 50 entries for illustration purposes:
vocab = {token:integer for integer,token in enumerate(all_words)} 

#for i, item in enumerate(vocab.items()):
#    print(item) 
#    if i > 50:
#        break

# We can use the tokenizer to encode (that is, tokenize) texts into integers
# These integers can then be embedded (later) as input of/for the LLM

tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

decoded_text = tokenizer.decode(ids)
print(decoded_text)

# It works only with words existing in the training set
# The problem is that the word "Hello" was not used in the The Verdict short story. 
# Hence, it is not contained in the vocabulary. 
# This highlights the need to consider large and diverse training sets to extend the vocabulary when working on LLMs.