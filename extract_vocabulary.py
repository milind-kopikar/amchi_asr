import json
import re

# Load test results
with open('nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/final_test_results.json') as f:
    data = json.load(f)

# Extract all reference texts
references = [sample['reference'] for sample in data['per_sample']]
print(f'Total samples: {len(references)}')

# Extract all words (remove punctuation and numbers)
all_words = set()
for ref in references:
    # Remove punctuation and numbers, split by spaces
    clean_ref = re.sub(r'[.,!?;:\"\'०-९]', '', ref)
    words = [w.strip() for w in clean_ref.split() if w.strip()]
    all_words.update(words)

# Sort alphabetically
unique_words = sorted(list(all_words))
print(f'Total unique words: {len(unique_words)}')

# Save to file for Gemini analysis
with open('deaf_speech_vocabulary.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(unique_words))

print('Vocabulary saved to deaf_speech_vocabulary.txt')
print('First 50 words:')
print('\n'.join(unique_words[:50]))