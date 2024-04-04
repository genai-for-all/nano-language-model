import re

with open("the-verdict.txt", "r", encoding="utf-8") as f: raw_text = f.read()
print("Total number of character:", len(raw_text)) 
print(raw_text[:99])

preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text) 
preprocessed = [item.strip() for item in preprocessed if item.strip()] 
print(len(preprocessed))

# Let's print the first 30 tokens for a quick visual check:
print(preprocessed[:30])
