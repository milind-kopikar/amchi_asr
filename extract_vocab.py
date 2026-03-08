import json

# Load test results
with open('nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/final_test_results.json') as f:
    data = json.load(f)

# Extract all reference texts
references = [sample['reference'] for sample in data['per_sample']]
print(f'Total samples: {len(references)}')

# Extract all words
all_words = set()
for ref in references:
    words = [w.strip('.,!?;:\"\'') for w in ref.split() if w.strip('.,!?;:\"\'')]
    all_words.update(words)

print(f'Total unique words: {len(all_words)}')
print('All words:')
print('\n'.join(sorted(list(all_words))))