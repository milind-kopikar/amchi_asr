const fs = require('fs');
const path = require('path');

const API_URL = 'https://konkani-dictionary-production.up.railway.app/api/dictionary';
const OUTPUT_FILE = path.join(__dirname, '../konkani_dictionary_full.json');

async function fetchAllEntries() {
    console.log('🚀 Starting dictionary download...');
    let allEntries = [];
    let page = 1;
    let hasMore = true;

    try {
        while (hasMore) {
            console.log(`📥 Fetching page ${page}...`);
            const response = await fetch(`${API_URL}?page=${page}&limit=100`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (!data.entries || data.entries.length === 0) {
                hasMore = false;
                break;
            }

            allEntries = allEntries.concat(data.entries);

            // Check if we've reached the last page
            if (page >= data.pagination.totalPages) {
                hasMore = false;
            } else {
                page++;
            }
        }

        console.log(`\n✅ Download complete!`);
        console.log(`📊 Total entries fetched: ${allEntries.length}`);

        // Save to file
        fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allEntries, null, 2));
        console.log(`💾 Saved to: ${OUTPUT_FILE}`);

        // Verify uniqueness
        const uniqueIds = new Set(allEntries.map(e => e.id));
        console.log(`🔍 Unique IDs: ${uniqueIds.size}`);

        if (uniqueIds.size !== allEntries.length) {
            console.warn('⚠️ Warning: Duplicate IDs detected!');
        }

    } catch (error) {
        console.error('❌ Error downloading dictionary:', error);
        process.exit(1);
    }
}

fetchAllEntries();
