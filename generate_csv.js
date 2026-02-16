const fs = require('fs');
const path = require('path');

const CANDIDATES_FILE = path.join(__dirname, '../story_candidates.json');
const OUTPUT_FILE = path.join(__dirname, '../community_review.csv');

function escapeCsv(text) {
    if (!text) return '';
    const str = String(text);
    if (str.includes(',') || str.includes('"') || str.includes('\n') || str.includes('\r')) {
        return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
}

async function generateCsv() {
    console.log('📊 Generating CSV...');

    if (!fs.existsSync(CANDIDATES_FILE)) {
        console.error('❌ Candidates file not found.');
        process.exit(1);
    }

    const candidates = JSON.parse(fs.readFileSync(CANDIDATES_FILE, 'utf8'));

    const headers = [
        'Action',
        'Konkani Word (Devanagari)',
        'Konkani Word (Roman/IAST)',
        'English Meaning',
        'Usage Example',
        'Usage Example (IAST)',
        'Notes'
    ];

    const rows = [headers.join(',')];

    candidates.forEach(c => {
        // For UPDATE, we might have existing values, but we want to confirm them
        // For ADD, we leave Roman/Meaning blank for community input

        // Attempt simple romanization placeholder if possible (optional)
        let roman = c.ai_iast || '';
        let meaning = c.ai_meaning || '';

        if (c.original_entry) {
            roman = roman || c.original_entry.word_konkani_english_alphabet || '';
            meaning = meaning || c.original_entry.english_meaning || '';
        }

        const row = [
            c.action,
            c.word, // Devanagari
            roman,  // IAST (ai_iast or original)
            meaning,// Meaning (ai_meaning or original)
            c.usage_context, // Usage
            c.ai_usage_iast || '', // Usage IAST
            c.notes
        ];

        rows.push(row.map(escapeCsv).join(','));
    });

    fs.writeFileSync(OUTPUT_FILE, rows.join('\r\n'));
    console.log(`✅ CSV generated with ${candidates.length} rows.`);
    console.log(`💾 Saved to: ${OUTPUT_FILE}`);
}

generateCsv();
