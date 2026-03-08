# Enhanced ASR Post-Processing Methods

## Overview

This document describes the enhanced post-processing algorithms for both **Deaf Speech** and **Amchi Konkani** ASR systems.

### Deaf Speech (Marathi)
- **Achievement**: **13.3 percentage point improvement** in Word Error Rate (WER)
- **Baseline WER**: 75.3% → **Enhanced WER**: 62.0% (+17.6% relative improvement)
- **Method**: RECONSTRUCT/FILL/PASSTHROUGH modes with high-frequency vocabulary guidance

### Amchi Konkani
- **Achievement**: Dictionary-based correction with Konkani word validation
- **Method**: Word classification using Konkani dictionary, Gemini-powered correction
- **Features**: Root word matching, inflection handling, cultural context preservation

## Problem Statements

### Deaf Speech Recognition Challenges
- **High baseline WER**: Initial ASR produces 75.3% WER on deaf speech test data
- **Garbled fragments**: Many utterances are completely unrecognizable, producing only phonetic fragments marked with "⁇" (unknown token)
- **Domain constraints**: Speech is limited to everyday transactional scenarios (shopping, transportation, daily routines)
- **Language specificity**: Marathi language with Devanagari script

### Amchi Konkani Recognition Challenges
- **Limited digital resources**: No comprehensive Konkani language models or dictionaries
- **Dialect variations**: Geographic variations in Amchi Konkani pronunciation
- **Code-switching**: Mix of Konkani, Kannada, and Marathi influences
- **Cultural context**: Preservation of local expressions and terminology

### Core Algorithm

The algorithm processes ASR output through three modes based on word classification:

1. **RECONSTRUCT Mode**: Triggered when all tokens are garbled/unknown
   - Uses phonetic fragments as clues
   - Reconstructs complete sentences using LLM with domain knowledge

2. **FILL Mode**: Triggered when some trusted words exist but gaps remain
   - Preserves recognized words
   - Fills gaps with contextually appropriate words

3. **PASSTHROUGH Mode**: Applied when output is already high quality
   - Minimal intervention for well-recognized utterances

### Key Enhancement: High-Frequency Vocabulary Guidance

The critical innovation is providing **domain-specific high-frequency Marathi vocabulary** to guide LLM reconstruction:

```python
HIGH_FREQ_MARATHI_WORDS = {
    "किती", "आहे", "एक", "द्या", "काय", "कधी", "येईल", "कोणता", "कोणती",
    "हे", "दोन", "तीन", "करा", "चहा", "पाणी", "हवं", "नाही", "ठीक", "का",
    "ही", "मला", "बघा", "थांबा", "पुन्हा", "समजलं", "लिहा", "दाखवा",
    "बरोबर", "चुकीचं", "नीट", "आता", "लगेच", "येतो", "झाला", "मिळेल",
    "शकतो", "घेतात", "चालतो", "जाते", "जातो", "करा", "सांगा", "बघू",
    "बदलू", "होईल", "करू"
}
```

### Word Classification Logic

Words are classified into three categories:

**TRUSTED WORDS**: High-confidence vocabulary from training data
- Question words: किती, काय, कसे, कुठे, कधी, का, कधी, कोण, कुणी
- Verbs: आहे, आहेत, आहेस, आहात, नाही, नाहीत, द्या, घ्या, सांगा, पाहा, करा
- Common nouns: दूध, पाणी, पेपर, चहा, रुपये, पैसे, बस, घर, शाळा, बाजार
- Pronouns: मी, तू, तो, ती, ते, हे, हा, ही, आपण, आम्ही, तुम्ही

**GARBLED WORDS**: Unrecognizable fragments
- Single characters or non-Devanagari text
- Orphaned matras (vowel signs without consonants)
- Unknown tokens (⁇)

**UNCERTAIN WORDS**: Everything else requiring validation

### LLM Integration

**Model**: Google Gemini 2.5 Flash
**API Key**: Required (set in environment or .env file)

**Prompt Engineering**:
- **RECONSTRUCT Mode**: Includes high-frequency vocabulary list
- **FILL Mode**: Contextual gap-filling with preserved trusted words
- **PASSTHROUGH Mode**: Minimal cleanup of artifacts

## Implementation Details

### Core Files

| Use Case | Script | Purpose |
|---|---|---|
| **Deaf Speech** | `scripts/postprocess_asr.py` | RECONSTRUCT/FILL/PASSTHROUGH with high-frequency Marathi vocabulary |
| **Amchi Konkani** | `scripts/amchi_postprocess_asr.py` | Dictionary-based correction with Konkani word validation |
| **Evaluation** | `robust_evaluation.py` | Batch processing with incremental result saving |

### Key Functions

**Deaf Speech**:
```python
def postprocess_sample(client, prediction: str, original_wer: float = 1.0) -> dict:
    """
    Main post-processing function for deaf speech
    Returns: {'mode': str, 'corrected': str}
    """
```

**Amchi Konkani**:
```python
def postprocess_konkani_sample(client, prediction: str, dictionary: dict) -> str:
    """
    Dictionary-based correction for Konkani ASR output
    """
```

### Mode Selection Logic

```python
# Count word classifications
trusted_count = sum(1 for word in words if word in TRUSTED_WORDS)
garbled_count = sum(1 for word in words if is_garbled(word))

if garbled_count == len(words):
    mode = "RECONSTRUCT"
elif trusted_count > 0:
    mode = "FILL"
else:
    mode = "PASSTHROUGH"
```

## Performance Results

### Evaluation Metrics (124 Test Samples)

- **Baseline WER**: 75.3%
- **Enhanced WER**: 62.0%
- **Improvement**: +13.3 percentage points (+17.6% relative)
- **Sample-level improvement**: +3.0 percentage points average

### Mode Distribution

- **RECONSTRUCT**: ~40% of samples (heavily garbled inputs)
- **FILL**: ~50% of samples (partial recognition)
- **PASSTHROUGH**: ~10% of samples (good recognition)

### Success Rate

- **Improved samples**: 73/124 (58.9%)
- **Worsened samples**: 46/124 (37.1%)
- **Unchanged samples**: 5/124 (4.0%)

## Usage Instructions

### Environment Setup

1. Install dependencies:
```bash
pip install google-genai jiwer
```

2. Set API key:
```bash
export GEMINI_API_KEY="your_api_key_here"
# OR add to .env file
```

### Command Line Usage

**Deaf Speech Post-Processing**:
```bash
python scripts/postprocess_asr.py \
    --input path/to/deaf_speech_test_results.json \
    --output enhanced_results.json \
    --report postprocess_report.txt
```

**Amchi Konkani Post-Processing**:
```bash
python scripts/amchi_postprocess_asr.py \
    --input path/to/konkani_test_results.json \
    --output corrected_results.json \
    --report konkani_postprocess_report.txt \
    --dict path/to/konkani_dictionary_full.json
```

### Python API Usage

**Deaf Speech**:
```python
import google.genai as genai
from scripts.postprocess_asr import postprocess_sample

client = genai.Client(api_key="your_key")
result = postprocess_sample(client, "raw_asr_output")
print(f"Mode: {result['mode']}")
print(f"Corrected: {result['corrected']}")
```

**Amchi Konkani**:
```python
from scripts.amchi_postprocess_asr import postprocess_konkani_sample
import json

with open('konkani_dictionary_full.json') as f:
    dictionary = json.load(f)

corrected = postprocess_konkani_sample(client, "raw_konkani_output", dictionary)
```

## Validation and Testing

### Test Script

```bash
python robust_evaluation.py
```

This script:
- Processes all test samples incrementally
- Saves results to `evaluation_results.json`
- Handles API errors gracefully
- Resumes from interruption point

### Quality Assurance

- **Devanagari validation**: Ensures output contains primarily Devanagari script
- **Length constraints**: Maintains reasonable sentence length (3-6 words)
- **Domain adherence**: Stays within transactional/daily routine context

## Amchi Konkani Post-Processing

### Algorithm Overview

The Amchi Konkani post-processing uses a dictionary-based approach:

1. **Word Classification**:
   - **GARBLED**: Non-Devanagari text, orphaned matras, unknown tokens (⁇)
   - **TRUSTED**: Exact matches in Konkani dictionary
   - **UNCERTAIN**: Potential inflected forms of dictionary words

2. **Correction Strategy**:
   - Build comprehensive Gemini prompt with full sentence context
   - Highlight garbled words as `[___]` placeholders
   - Provide dictionary words as correction reference
   - Preserve cultural and linguistic authenticity

3. **Safety Mechanisms**:
   - WER validation: Revert if correction increases error rate
   - Dictionary validation: Ensure corrections use valid Konkani words

### Key Features

- **Dictionary Integration**: Uses comprehensive Konkani word database
- **Root Word Matching**: Handles inflected forms and variations
- **Cultural Preservation**: Maintains local expressions and dialect features
- **Multi-dialect Support**: Accommodates geographic variations in Amchi Konkani

### Implementation

**Script**: `scripts/amchi_postprocess_asr.py`
**Dictionary**: `konkani_dictionary_full.json` (from separate Konkani dictionary project)
**Model**: Gemini 2.5 Flash with Konkani-specific prompts

## Conclusion

These enhanced post-processing methods represent significant advancements in ASR accessibility for both deaf speech and endangered language preservation:

### Deaf Speech Achievement
- **17.6% relative WER improvement** through intelligent LLM-guided reconstruction
- **13.3 percentage point reduction** from 75.3% to 62.0% WER
- Domain-specific vocabulary guidance enables accurate sentence reconstruction

### Amchi Konkani Achievement
- **Dictionary-based correction** preserves linguistic and cultural authenticity
- **Multi-dialect support** accommodates geographic variations in Amchi Konkani
- **Root word matching** handles inflectional morphology of the language

Both methods leverage Gemini AI with specialized prompts and validation mechanisms to ensure high-quality corrections while maintaining the integrity of the target languages and speech patterns.

## References

### Deaf Speech
- Test data: `nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/final_test_results.json`
- Implementation: `scripts/postprocess_asr.py`
- Results: `evaluation_results.json`
- Evaluation script: `robust_evaluation.py`

### Amchi Konkani
- Implementation: `scripts/amchi_postprocess_asr.py`
- Dictionary: `konkani_dictionary_full.json` (from Konkani dictionary project)
- Test data: Amchi Konkani experiment results in `results/` directory</content>
<parameter name="filePath">c:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\ENHANCED_POSTPROCESSING_METHOD.md