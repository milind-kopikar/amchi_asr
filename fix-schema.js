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

async function fixSchema() {
  try {
    console.log('🔧 Fixing database schema to allow NULL English meanings...');
    
    // Remove NOT NULL constraint from english_meaning
    await pool.query('ALTER TABLE dictionary_entries ALTER COLUMN english_meaning DROP NOT NULL');
    
    console.log('✅ Schema updated successfully!');
    
    // Show current stats
    const statsQuery = `
      SELECT 
        COUNT(*) as total_entries,
        COUNT(english_meaning) as with_english_meaning,
        COUNT(*) - COUNT(english_meaning) as without_english_meaning
      FROM dictionary_entries
    `;
    
    const result = await pool.query(statsQuery);
    const stats = result.rows[0];
    
    console.log('\n📊 Current database stats:');
    console.log(`Total entries: ${stats.total_entries}`);
    console.log(`With English meaning: ${stats.with_english_meaning}`);
    console.log(`Without English meaning: ${stats.without_english_meaning}`);
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await pool.end();
  }
}

fixSchema();