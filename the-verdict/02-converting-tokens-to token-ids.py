# We build a vocabulary by tokenizing the entire text in a training dataset into individual tokens. 
# These individual tokens are then sorted alphabetically, and duplicate tokens are removed. 
# The unique tokens are then aggregated into a vocabulary that defines a mapping from each unique token to a unique integer value. 
#The depicted vocabulary is purposefully small for illustration purposes and contains no punctuation or special characters for simplicity.

import re

with open("the-verdict.txt", "r", encoding="utf-8") as f: raw_text = f.read()
print("Total number of character:", len(raw_text)) 
print(raw_text[:99])

preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text) 
preprocessed = [item.strip() for item in preprocessed if item.strip()] 
print(len(preprocessed))

# Let's print the first 30 tokens for a quick visual check:
print(preprocessed[:30])

all_words = sorted(list(set(preprocessed))) 
vocab_size = len(all_words) 
print(vocab_size)

# After determining that the vocabulary size is 1,159 via the above code, 
# we create the vocabulary and print its first 50 entries for illustration purposes:
vocab = {token:integer for integer,token in enumerate(all_words)} 
for i, item in enumerate(vocab.items()):
    print(item) 
    if i > 50:
        break