const fs = require('fs');
const path = require('path');

const DICT_FILE = path.join(__dirname, '../konkani_dictionary_full.json');
const CANDIDATES_FILE = path.join(__dirname, '../story_candidates.json');

// Get story file from command line argument or default to story1.txt
const storyFileName = process.argv[2] || 'story1.txt';
const STORY_FILE = path.join(__dirname, `../${storyFileName}`);

// Simple punctuation removal for Konkani/Devanagari
function cleanWord(word) {
    // Include Devanagari Danda \u0964 in the removal list
    return word.replace(/[|!?,."()\-–\[\]\u0964]/g, '').trim();
}

// Check if a word contains Devanagari characters
function isDevanagari(word) {
    const devanagariRange = /[\u0900-\u097F]/;
    return devanagariRange.test(word);
}

async function processStory() {
    console.log(`📖 Reading ${storyFileName}...`);

    if (!fs.existsSync(DICT_FILE)) {
        console.error('❌ Dictionary file not found. Run download_dictionary.js first.');
        process.exit(1);
    }

    const dictionary = JSON.parse(fs.readFileSync(DICT_FILE, 'utf8'));
    const storyText = fs.readFileSync(STORY_FILE, 'utf8');

    // Load existing candidates to merge/append
    let candidates = [];
    if (fs.existsSync(CANDIDATES_FILE)) {
        candidates = JSON.parse(fs.readFileSync(CANDIDATES_FILE, 'utf8'));
        console.log(`📚 Loaded ${candidates.length} existing candidates.`);
    }

    // Create a map of existing Devanagari words for fast lookup
    // Normalize by trimming and lowercase (though Devanagari doesn't have case, good practice)
    const existingWords = new Set();
    const wordMap = new Map(); // entry -> full object

    dictionary.forEach(entry => {
        if (entry.word_konkani_devanagari) {
            const clean = cleanWord(entry.word_konkani_devanagari);
            existingWords.add(clean);
            wordMap.set(clean, entry);
        }
    });

    console.log(`📚 Loaded ${existingWords.size} unique Devanagari words from dictionary.`);

    // Split story into sentences first to capture context
    // Splitting by | (danda) or ! or ? or . or \u0964 (Devanagari danda) or Newlines
    const sentences = storyText.split(/[|!?.\u0964\r\n]+/).map(s => s.trim()).filter(s => s.length > 0);

    const processedWords = new Set(candidates.map(c => c.word)); // Track what we already have

    console.log(`🧩 Processing ${sentences.length} sentences...`);

    let newCandidatesCount = 0;

    sentences.forEach(sentence => {
        const words = sentence.split(/\s+/);

        words.forEach(rawWord => {
            const word = cleanWord(rawWord);

            // Skip empty, non-Devanagari, or single char words (unless specific like 'न')
            if (!word || !isDevanagari(word) || word.length < 2) return;

            // Skip if we already processed this word (either from prev run or curr run)
            if (processedWords.has(word)) return;
            processedWords.add(word);

            const exists = existingWords.has(word);
            let action = 'ADD';
            let notes = '';
            let originalEntry = null;

            if (exists) {
                originalEntry = wordMap.get(word);

                // Check if we can augment
                let needsUpdate = false;
                if (!originalEntry.english_meaning) {
                    notes += 'Missing Meaning. ';
                    needsUpdate = true;
                }
                if (!originalEntry.context_usage_sentence) {
                    notes += 'Missing Usage. ';
                    needsUpdate = true;
                }

                if (needsUpdate) {
                    action = 'UPDATE';
                } else {
                    action = 'SKIP'; // Already comprehensive
                }
            }

            if (action !== 'SKIP') {
                candidates.push({
                    word: word,
                    action: action,
                    usage_context: sentence, // The full sentence where it was found
                    original_entry: originalEntry,
                    notes: notes.trim()
                });
                newCandidatesCount++;
            }
        });
    });

    console.log(`✅ Processing complete!`);
    console.log(`📋 Total candidates now: ${candidates.length} (New added: ${newCandidatesCount})`);

    fs.writeFileSync(CANDIDATES_FILE, JSON.stringify(candidates, null, 2));
    console.log(`💾 Saved merged candidates to: ${CANDIDATES_FILE}`);
}

processStory();
