const fs = require('fs');
const csv = require('csv-parser');
const { Pool } = require('pg');
require('dotenv').config();

/**
 * Konkani Dictionary Spreadsheet Import Utility
 * 
 * This utility imports your spreadsheet data into PostgreSQL
 * Matches your exact column structure from the spreadsheet
 */

// Database connection
const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'konkani_dictionary',
  user: process.env.DB_USER || 'konkani_dev',
  password: process.env.DB_PASSWORD,
  // Ensure proper Unicode handling
  client_encoding: 'UTF8',
  application_name: 'konkani_dictionary_import'
});

/**
 * Import CSV file to database
 * @param {string} csvFilePath - Path to your exported CSV file
 */
async function importSpreadsheetData(csvFilePath) {
  const client = await pool.connect();
  
  try {
    console.log('🚀 Starting import of Konkani dictionary data...');
    console.log(`📁 Reading file: ${csvFilePath}`);
    
    // Start transaction
    await client.query('BEGIN');
    
    let importCount = 0;
    let errorCount = 0;
    let batchSize = 100;
    let currentBatch = 0;
    
    // Read and process CSV
    const results = [];
    
    return new Promise((resolve, reject) => {
      fs.createReadStream(csvFilePath, { encoding: 'utf8' })
        .pipe(csv({
          // Configuration for proper Unicode/diacritical handling
          skipEmptyLines: true,
          skipLinesWithError: true, // Skip malformed lines instead of failing
          separator: ',',
          quote: '"',
          escape: '"',
          // Map your exact column headers to our database fields  
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
          
          for (const row of results) {
            try {
              // Clean and validate data
              const entryData = {
                entry_number: parseInt(row.entry_number) || null,
                word_konkani_devanagari: cleanText(row.word_konkani_devanagari),
                word_konkani_english_alphabet: cleanText(row.word_konkani_english_alphabet),
                english_meaning: cleanText(row.english_meaning),
                context_usage_sentence: cleanText(row.context_usage_sentence),
                devanagari_needs_correction: parseYesNo(row.devanagari_needs_correction),
                meaning_needs_correction: parseYesNo(row.meaning_needs_correction),
                corrected_devanagari_user_x: cleanText(row.corrected_devanagari_user_x),
                corrected_devanagari_user_y: cleanText(row.corrected_devanagari_user_y),
                corrected_meaning_user_x: cleanText(row.corrected_meaning_user_x),
                corrected_meaning_user_y: cleanText(row.corrected_meaning_user_y)
              };
              
              // Skip empty rows - need either English alphabet OR Devanagari + meaning
              if (!entryData.word_konkani_english_alphabet && !entryData.word_konkani_devanagari && !entryData.english_meaning) {
                continue;
              }
              
              // Skip entries with no meaningful content
              if (!entryData.word_konkani_english_alphabet?.trim() && 
                  !entryData.word_konkani_devanagari?.trim() && 
                  !entryData.english_meaning?.trim()) {
                continue;
              }
              
              // If no English alphabet, use Devanagari as placeholder or vice versa
              if (!entryData.word_konkani_english_alphabet && entryData.word_konkani_devanagari) {
                entryData.word_konkani_english_alphabet = `[${entryData.word_konkani_devanagari}]`;
              }
              
              // Insert into database with conflict handling
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
              
              await client.query(insertQuery, [
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
              
              importCount++;
              
              if (importCount % 50 === 0) {
                console.log(`✅ Imported ${importCount} entries...`);
                // Commit batch periodically to avoid large transaction rollbacks
                await client.query('COMMIT');
                await client.query('BEGIN');
              }
              
            } catch (error) {
              console.error(`❌ Error importing row ${row.entry_number || 'unknown'}:`, error.message);
              errorCount++;
            }
          }
          
          // Commit transaction
          await client.query('COMMIT');
          
          console.log(`\n🎉 Import completed!`);
          console.log(`✅ Successfully imported: ${importCount} entries`);
          console.log(`❌ Errors: ${errorCount}`);
          console.log(`📊 Total processed: ${results.length} rows`);
          
          // Generate import summary
          const summary = await generateImportSummary(client);
          console.log('\n📈 Database Summary:');
          console.log(summary);
          
          resolve({ imported: importCount, errors: errorCount, total: results.length });
        })
        .on('error', (error) => {
          reject(error);
        });
    });
    
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Import failed:', error);
    throw error;
  } finally {
    client.release();
  }
}

/**
 * Helper functions
 */
function cleanText(text) {
  if (!text) return null;
  
  // Convert to string and normalize Unicode (important for diacriticals)
  let cleaned = text.toString().normalize('NFC').trim();
  
  // Return null for empty strings, preserve diacritical marks
  return cleaned === '' ? null : cleaned;
}

function parseYesNo(value) {
  if (!value) return false;
  const cleaned = value.toString().trim().toLowerCase();
  return cleaned === 'y' || cleaned === 'yes' || cleaned === '1' || cleaned === 'true';
}

async function generateImportSummary(client) {
  const queries = [
    { label: 'Total entries', query: 'SELECT COUNT(*) as count FROM dictionary_entries' },
    { label: 'Entries with Devanagari', query: 'SELECT COUNT(*) as count FROM dictionary_entries WHERE word_konkani_devanagari IS NOT NULL' },
    { label: 'Entries needing correction', query: 'SELECT COUNT(*) as count FROM dictionary_entries WHERE devanagari_needs_correction = true OR meaning_needs_correction = true' },
    { label: 'Entries with user corrections', query: 'SELECT COUNT(*) as count FROM dictionary_entries WHERE corrected_devanagari_user_x IS NOT NULL OR corrected_meaning_user_x IS NOT NULL' }
  ];
  
  let summary = '';
  for (const { label, query } of queries) {
    const result = await client.query(query);
    summary += `${label}: ${result.rows[0].count}\n`;
  }
  
  return summary;
}

/**
 * Command line usage
 */
if (require.main === module) {
  const csvFilePath = process.argv[2];
  
  if (!csvFilePath) {
    console.log('Usage: node import-spreadsheet.js <path-to-csv-file>');
    console.log('Example: node import-spreadsheet.js ./konkani-dictionary.csv');
    process.exit(1);
  }
  
  if (!fs.existsSync(csvFilePath)) {
    console.error(`❌ File not found: ${csvFilePath}`);
    process.exit(1);
  }
  
  importSpreadsheetData(csvFilePath)
    .then(() => {
      console.log('✅ Import completed successfully!');
      process.exit(0);
    })
    .catch((error) => {
      console.error('❌ Import failed:', error);
      process.exit(1);
    });
}

module.exports = { importSpreadsheetData };