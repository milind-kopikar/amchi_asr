import os
import sys
import sentencepiece as spm

# Search for a .model file in likely locations
model_path = None
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.model') and ('tokenizer' in root.lower() or 'tokenizers' in root.lower() or 'token' in file.lower()):
            model_path = os.path.join(root, file)
            break
    if model_path:
        break

# fallback: search in models/tokenizer
if not model_path:
    for root, dirs, files in os.walk('models'):
        for file in files:
            if file.endswith('.model'):
                model_path = os.path.join(root, file)
                break
        if model_path:
            break

if not model_path:
    print('CRITICAL ERROR: Could not find .model file in tokenizers/ or models/ directory.')
    sys.exit(1)

print(f"DEBUG: Loading tokenizer from {model_path}")
sp = spm.SentencePieceProcessor(model_file=model_path)

# Test case
text = "रोहन होड ज़ाल्लो!"
print(f"DEBUG: Input Text: '{text}'")
ids = sp.encode(text, out_type=int)
print(f"DEBUG: Encoded IDs: {ids}")

unk_id = sp.unk_id()
print(f"DEBUG: UNK id: {unk_id}")

if unk_id in ids:
    print(f"FAIL: The text contains UNK tokens (ID {unk_id}). The tokenizer cannot read this text.")
    for ch in text:
        ch_id = sp.encode(ch, out_type=int)
        if unk_id in ch_id:
            print(f" -> Character '{ch}' (U+{ord(ch):04X}) maps to UNK")
else:
    print("PASS: The tokenizer successfully encoded the text without UNKs.")

# Round-trip decode
decoded = sp.decode(ids)
print(f"DEBUG: Round-trip Decode: '{decoded}'")
