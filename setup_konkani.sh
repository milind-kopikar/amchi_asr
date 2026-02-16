#!/bin/bash
set -e

echo "🚀 Setting up Konkani ASR Model and Tokenizer..."

# 1. Create directories
mkdir -p models tokenizers

# 2. Download Konkani Model (AI4Bharat IndicConformer Hybrid)
if [ ! -f "models/konkani_model.nemo" ]; then
    echo "📥 Downloading Konkani model from Hugging Face..."
    wget https://huggingface.co/ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large/resolve/main/indicconformer_stt_kok_hybrid_ctc_rnnt_large.nemo \
        -O models/konkani_model.nemo
else
    echo "✅ Konkani model already exists."
fi

# 3. Extract the CORRECT Konkani tokenizer from the .nemo archive
# The Hugging Face 'tokenizer.model' file is often generic or missing Devanagari.
# We extract the specific one used for Konkani (kok) from the multilingual bundle.
echo "📦 Extracting Konkani-specific tokenizer from .nemo archive..."

# Hash for Konkani BPE tokenizer in this specific model version
TOKENIZER_HASH="def9dd6f2f9b4f5fb30c152c456a65cd"
VOCAB_HASH="15a253cba9a84206a72745fe5615bdc4"

tar -xf models/konkani_model.nemo "./${TOKENIZER_HASH}_tokenizer.model" -C tokenizers/
tar -xf models/konkani_model.nemo "./${VOCAB_HASH}_vocab.txt" -C tokenizers/

# 4. Rename to standard names for our configs
mv "tokenizers/${TOKENIZER_HASH}_tokenizer.model" tokenizers/konkani_tokenizer.model
mv "tokenizers/${VOCAB_HASH}_vocab.txt" tokenizers/vocab.txt

echo "✅ Konkani setup complete!"
echo "Model: models/konkani_model.nemo"
echo "Tokenizer: tokenizers/konkani_tokenizer.model"
echo "Vocab: tokenizers/vocab.txt"
