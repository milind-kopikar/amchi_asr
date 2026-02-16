const fs = require('fs');
const csv = require('csv-parser');
const { Pool } = require('pg');
require('dotenv').config();

/**
 * Enhanced Konkani Dictionary Import with Better Error Handling
 * 
 * This version:
 * - Processes each row individually
 * - Skips problematic rows instead of failing
 * - Provides detailed error logging
 * - Handles encoding issues better
 */

// Database connection
const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'konkani_dictionary',
  user: process.env.DB_USER || 'konkani_dev',
  password: process.env.DB_PASSWORD,
  client_encoding: 'UTF8',
  application_name: 'konkani_dictionary_import_v2'
});

/**
 * Clean text with better Unicode handling
 */
function cleanText(text) {
  if (!text || typeof text !== 'string') return null;
  
  // Normalize Unicode (NFC - Canonical Decomposition, followed by Canonical Composition)
  let cleaned = text.normalize('NFC');
  
  // Remove BOM and other invisible characters
  cleaned = cleaned.replace(/^\uFEFF/, ''); // Remove BOM
  cleaned = cleaned.replace(/[\u200B-\u200D\uFEFF]/g, ''); // Remove zero-width chars
  
  // Trim whitespace
  cleaned = cleaned.trim();
  
  // Return null for empty strings
  return cleaned.length > 0 ? cleaned : null;
}

/**
 * Parse Y/N values more robustly
 */
function parseYesNo(value) {
  if (!value || typeof value !== 'string') return false;
  const normalized = value.toLowerCase().trim();
  return ['y', 'yes', 'true', '1'].includes(normalized);
}

/**
 * Validate an entry before insertion
 */
function validateEntry(entry, rowIndex) {
  const errors = [];
  
  // Must have at least one of these
  if (!entry.word_konkani_devanagari && !entry.word_konkani_english_alphabet && !entry.english_meaning) {
    errors.push('Missing all key fields (Devanagari, English alphabet, and meaning)');
  }
  
  // Check for obviously corrupted data
  if (entry.word_konkani_devanagari && entry.word_konkani_devanagari.includes('à¤')) {
    errors.push('Corrupted UTF-8 encoding in Devanagari field');
  }
  
  if (entry.word_konkani_english_alphabet && entry.word_konkani_english_alphabet.includes('à¤')) {
    errors.push('Corrupted UTF-8 encoding in English alphabet field');
  }
  
  // Check for reasonable lengths
  if (entry.english_meaning && entry.english_meaning.length > 500) {
    errors.push('English meaning too long (>500 chars)');
  }
  
  return errors;
}

/**
 * Import single entry with individual error handling
 */
async function importSingleEntry(client, entry, rowIndex) {
  try {
    // Validate entry
    const validationErrors = validateEntry(entry, rowIndex);
    if (validationErrors.length > 0) {
      return { success: false, error: `Validation failed: ${validationErrors.join(', ')}` };
    }
    
    // Prepare data
    const entryData = {
      entry_number: parseInt(entry.entry_number) || null,
      word_konkani_devanagari: cleanText(entry.word_konkani_devanagari),
      word_konkani_english_alphabet: cleanText(entry.word_konkani_english_alphabet),
      english_meaning: cleanText(entry.english_meaning),
      context_usage_sentence: cleanText(entry.context_usage_sentence),
      devanagari_needs_correction: parseYesNo(entry.devanagari_needs_correction),
      meaning_needs_correction: parseYesNo(entry.meaning_needs_correction),
      corrected_devanagari_user_x: cleanText(entry.corrected_devanagari_user_x),
      corrected_devanagari_user_y: cleanText(entry.corrected_devanagari_user_y),
      corrected_meaning_user_x: cleanText(entry.corrected_meaning_user_x),
      corrected_meaning_user_y: cleanText(entry.corrected_meaning_user_y)
    };
    
    // Auto-fill missing English alphabet with Devanagari in brackets
    if (!entryData.word_konkani_english_alphabet && entryData.word_konkani_devanagari) {
      entryData.word_konkani_english_alphabet = `[${entryData.word_konkani_devanagari}]`;
    }
    
    // Insert query with individual transaction
    const insertQuery = `
      INSERT INTO dictionary_entries (
        entry_number,
        word_konkani_devanagari,
        word_konkani_english_alphabet,
        english_meaning,
        context_usage_sentence,
        devanagari_needs_correction,
        meaning_needs_correction,
        corrected_devanagari_user_x,
        corrected_devanagari_user_y,
        corrected_meaning_user_x,
        corrected_meaning_user_y
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
      ON CONFLICT (entry_number) DO UPDATE SET
        word_konkani_devanagari = EXCLUDED.word_konkani_devanagari,
        word_konkani_english_alphabet = EXCLUDED.word_konkani_english_alphabet,
        english_meaning = EXCLUDED.english_meaning,
        context_usage_sentence = EXCLUDED.context_usage_sentence,
        devanagari_needs_correction = EXCLUDED.devanagari_needs_correction,
        meaning_needs_correction = EXCLUDED.meaning_needs_correction,
        corrected_devanagari_user_x = EXCLUDED.corrected_devanagari_user_x,
        corrected_devanagari_user_y = EXCLUDED.corrected_devanagari_user_y,
        corrected_meaning_user_x = EXCLUDED.corrected_meaning_user_x,
        corrected_meaning_user_y = EXCLUDED.corrected_meaning_user_y,
        updated_at = CURRENT_TIMESTAMP
    `;
    
    const result = await client.query(insertQuery, [
      entryData.entry_number,
      entryData.word_konkani_devanagari,
      entryData.word_konkani_english_alphabet,
      entryData.english_meaning,
      entryData.context_usage_sentence,
      entryData.devanagari_needs_correction,
      entryData.meaning_needs_correction,
      entryData.corrected_devanagari_user_x,
      entryData.corrected_devanagari_user_y,
      entryData.corrected_meaning_user_x,
      entryData.corrected_meaning_user_y
    ]);
    
    return { success: true, data: entryData };
    
  } catch (error) {
    return { success: false, error: error.message, data: entry };
  }
}

/**
 * Enhanced import function with better error handling
 */
async function importSpreadsheetDataEnhanced(csvFilePath) {
  console.log('🚀 Starting enhanced import of Konkani dictionary data...');
  console.log(`📁 Reading file: ${csvFilePath}`);
  
  let importCount = 0;
  let errorCount = 0;
  let skippedCount = 0;
  const errors = [];
  
  return new Promise((resolve, reject) => {
    const results = [];
    
    fs.createReadStream(csvFilePath, { encoding: 'utf8' })
      .pipe(csv({
        skipEmptyLines: true,
        skipLinesWithError: false, // We'll handle errors manually
        separator: ',',
        quote: '"',
        escape: '"',
        mapHeaders: ({ header }) => {
          const headerMappings = {
            'Number': 'entry_number',
            'Konkani Word in Devnagiri': 'word_konkani_devanagari',
            'Konkani word in English alphabet': 'word_konkani_english_alphabet',
            'English Meaning': 'english_meaning',
            'Context/usage in a sentence': 'context_usage_sentence',
            'Does the Devnagiri Spelling need correction? (Y/N)': 'devanagari_needs_correction',
            'Does the meaning need correction? (Y/N)': 'meaning_needs_correction',
            'Corrected Devnagiri Spelling by user X': 'corrected_devanagari_user_x',
            'Corrected Devnagiri Spelling by user Y': 'corrected_devanagari_user_y',
            'Corrected English meaning by user X': 'corrected_meaning_user_x',
            'Corrected English meaning by user Y': 'corrected_meaning_user_y'
          };
          
          return headerMappings[header] || header.toLowerCase().replace(/[^a-z0-9]/g, '_');
        }
      }))
      .on('data', (row) => {
        results.push(row);
      })
      .on('end', async () => {
        console.log(`📊 Processing ${results.length} entries from CSV...`);
        
        // Process each row individually
        for (let i = 0; i < results.length; i++) {
          const row = results[i];
          const rowIndex = i + 1;
          
          // Skip completely empty rows
          const hasContent = Object.values(row).some(value => value && value.toString().trim());
          if (!hasContent) {
            skippedCount++;
            continue;
          }
          
          // Get database client for this row
          const client = await pool.connect();
          
          try {
            const result = await importSingleEntry(client, row, rowIndex);
            
            if (result.success) {
              importCount++;
              if (importCount % 100 === 0) {
                console.log(`✅ Imported ${importCount} entries (Row ${rowIndex}/${results.length})`);
              }
            } else {
              errorCount++;
              errors.push({
                row: rowIndex,
                entry_number: row.entry_number,
                error: result.error,
                data: result.data
              });
              
              if (errorCount % 50 === 0) {
                console.log(`❌ ${errorCount} errors so far (Row ${rowIndex}/${results.length})`);
              }
            }
            
          } catch (error) {
            errorCount++;
            errors.push({
              row: rowIndex,
              entry_number: row.entry_number,
              error: error.message,
              data: row
            });
          } finally {
            client.release();
          }
        }
        
        // Print summary
        console.log(`\n🎉 Enhanced import completed!`);
        console.log(`✅ Successfully imported: ${importCount} entries`);
        console.log(`❌ Errors: ${errorCount}`);
        console.log(`⏭️ Skipped empty: ${skippedCount}`);
        console.log(`📊 Total processed: ${results.length} rows`);
        
        // Show sample errors
        if (errors.length > 0) {
          console.log(`\n🔍 Sample errors (first 10):`);
          errors.slice(0, 10).forEach(err => {
            console.log(`Row ${err.row} (Entry #${err.entry_number || 'N/A'}): ${err.error}`);
          });
          
          // Save detailed error log
          const errorLogPath = 'import-errors.json';
          fs.writeFileSync(errorLogPath, JSON.stringify(errors, null, 2));
          console.log(`\n📝 Detailed error log saved to: ${errorLogPath}`);
        }
        
        resolve({ imported: importCount, errors: errorCount, skipped: skippedCount, total: results.length });
      })
      .on('error', (error) => {
        reject(error);
      });
  });
}

// Main execution
async function main() {
  try {
    const csvFilePath = process.argv[2] || './data/konkani_dictionary.csv';
    
    if (!fs.existsSync(csvFilePath)) {
      console.error(`❌ CSV file not found: ${csvFilePath}`);
      console.log('Usage: node enhanced-import.js <path-to-csv-file>');
      process.exit(1);
    }
    
    const result = await importSpreadsheetDataEnhanced(csvFilePath);
    
    console.log('\n✨ Import process completed successfully!');
    console.log('You can now test your dictionary with the updated data.');
    
  } catch (error) {
    console.error('❌ Import failed:', error);
  } finally {
    await pool.end();
  }
}

if (require.main === module) {
  main();
}

module.exports = { importSpreadsheetDataEnhanced };