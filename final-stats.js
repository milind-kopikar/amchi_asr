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

async function getFinalStats() {
  try {
    console.log('📊 Getting final database statistics...\n');
    
    // Overall stats
    const overallQuery = `
      SELECT 
        COUNT(*) as total_entries,
        COUNT(word_konkani_devanagari) as with_devanagari,
        COUNT(word_konkani_english_alphabet) as with_english_alphabet,
        COUNT(english_meaning) as with_english_meaning,
        COUNT(context_usage_sentence) as with_context,
        MIN(entry_number) as min_entry,
        MAX(entry_number) as max_entry
      FROM dictionary_entries
    `;
    
    const overallResult = await pool.query(overallQuery);
    const stats = overallResult.rows[0];
    
    console.log('🎯 FINAL DICTIONARY STATISTICS:');
    console.log('===============================');
    console.log(`📚 Total entries: ${stats.total_entries}`);
    console.log(`🕉️ With Devanagari: ${stats.with_devanagari} (${(stats.with_devanagari/stats.total_entries*100).toFixed(1)}%)`);
    console.log(`🔤 With English alphabet: ${stats.with_english_alphabet} (${(stats.with_english_alphabet/stats.total_entries*100).toFixed(1)}%)`);
    console.log(`🇬🇧 With English meaning: ${stats.with_english_meaning} (${(stats.with_english_meaning/stats.total_entries*100).toFixed(1)}%)`);
    console.log(`📝 With context/usage: ${stats.with_context} (${(stats.with_context/stats.total_entries*100).toFixed(1)}%)`);
    console.log(`📊 Entry range: #${stats.min_entry} to #${stats.max_entry}`);
    
    // Sample entries
    console.log('\n🔍 SAMPLE ENTRIES:');
    console.log('=================');
    const sampleQuery = `
      SELECT entry_number, word_konkani_devanagari, word_konkani_english_alphabet, english_meaning 
      FROM dictionary_entries 
      WHERE word_konkani_devanagari IS NOT NULL 
      ORDER BY RANDOM() 
      LIMIT 5
    `;
    
    const sampleResult = await pool.query(sampleQuery);
    sampleResult.rows.forEach(row => {
      console.log(`#${row.entry_number}: ${row.word_konkani_devanagari} | ${row.word_konkani_english_alphabet || 'N/A'} | ${row.english_meaning || 'N/A'}`);
    });
    
    // Words needing correction
    const correctionQuery = `
      SELECT 
        COUNT(*) FILTER (WHERE devanagari_needs_correction = true) as devanagari_corrections,
        COUNT(*) FILTER (WHERE meaning_needs_correction = true) as meaning_corrections
      FROM dictionary_entries
    `;
    
    const correctionResult = await pool.query(correctionQuery);
    const corrections = correctionResult.rows[0];
    
    console.log('\n⚠️ ENTRIES NEEDING CORRECTION:');
    console.log('=============================');
    console.log(`🕉️ Devanagari corrections needed: ${corrections.devanagari_corrections}`);
    console.log(`🇬🇧 Meaning corrections needed: ${corrections.meaning_corrections}`);
    
    console.log('\n✨ SUCCESS! Your complete Konkani dictionary is now ready!');
    console.log('🌐 Refresh your browser at http://localhost:3002 to explore all entries!');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await pool.end();
  }
}

getFinalStats();