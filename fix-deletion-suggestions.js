const { Pool } = require('pg');
require('dotenv').config();

const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'konkani_dictionary',
  user: process.env.DB_USER || 'konkani_dev',
  password: process.env.DB_PASSWORD,
  client_encoding: 'UTF8'
});

async function fixDeletionSuggestions() {
  try {
    console.log('🔧 Fixing existing deletion suggestions to populate original entry data...');

    // Find all deletion suggestions that have original_entry_id but missing original data
    const findQuery = `
      SELECT s.id, s.original_entry_id, e.word_konkani_devanagari, e.word_konkani_english_alphabet, e.english_meaning, e.context_usage_sentence
      FROM dictionary_suggestions s
      JOIN dictionary_entries e ON s.original_entry_id = e.id
      WHERE s.suggestion_type = 'deletion'
      AND s.original_entry_id IS NOT NULL
      AND (s.original_word_konkani_devanagari IS NULL OR s.original_word_konkani_devanagari = '')
    `;

    const result = await pool.query(findQuery);
    const suggestionsToFix = result.rows;

    console.log(`Found ${suggestionsToFix.length} deletion suggestions that need fixing`);

    if (suggestionsToFix.length === 0) {
      console.log('✅ No deletion suggestions need fixing!');
      return;
    }

    // Update each suggestion with the original entry data
    for (const suggestion of suggestionsToFix) {
      console.log(`Updating suggestion ${suggestion.id}...`);

      await pool.query(`
        UPDATE dictionary_suggestions
        SET
          original_word_konkani_devanagari = $1,
          original_word_konkani_english_alphabet = $2,
          original_english_meaning = $3,
          original_context_usage_sentence = $4
        WHERE id = $5
      `, [
        suggestion.word_konkani_devanagari,
        suggestion.word_konkani_english_alphabet,
        suggestion.english_meaning,
        suggestion.context_usage_sentence,
        suggestion.id
      ]);
    }

    console.log('✅ All deletion suggestions updated successfully!');

    // Show updated suggestions
    const verifyQuery = `
      SELECT id, original_word_konkani_devanagari, original_word_konkani_english_alphabet, original_english_meaning
      FROM dictionary_suggestions
      WHERE suggestion_type = 'deletion'
      AND original_entry_id IS NOT NULL
    `;

    const verifyResult = await pool.query(verifyQuery);
    console.log('\n📊 Updated deletion suggestions:');
    verifyResult.rows.forEach(row => {
      console.log(`ID: ${row.id}`);
      console.log(`  Devanagari: ${row.original_word_konkani_devanagari || 'N/A'}`);
      console.log(`  English: ${row.original_word_konkani_english_alphabet || 'N/A'}`);
      console.log(`  Meaning: ${row.original_english_meaning || 'N/A'}`);
      console.log('');
    });

  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await pool.end();
  }
}

fixDeletionSuggestions();