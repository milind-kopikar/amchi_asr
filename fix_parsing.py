import os
import re

def split_sentences(text):
    sentences = []
    current = ""
    i = 0
    while i < len(text):
        char = text[i]
        current += char
        if char in ['।', '!', '?']:
            # check if next is "
            if i + 1 < len(text) and text[i + 1] == '"':
                current += '"'
                i += 1
            sentences.append(current.strip())
            current = ""
        i += 1
    if current.strip():
        sentences.append(current.strip())
    return sentences

# Read the story1.txt
with open('story1.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Split into sentences
sentences = split_sentences(text)

print(f"Number of sentences: {len(sentences)}")

# Create directory if not exists
os.makedirs('story1_corpus/text_sentences', exist_ok=True)

# Write each sentence to a file
for i, sentence in enumerate(sentences, 1):
    filename = f'story1_corpus/text_sentences/story1_sentence{i}.txt'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(sentence)

print("Sentences written to files.")