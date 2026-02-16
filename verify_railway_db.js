/**
 * Verify connection to Railway PostgreSQL and show dictionary_entries structure.
 * 
 * Usage:
 *   Set DATABASE_URL in .env (from Railway dashboard: Variables → DATABASE_URL)
 *   Or: DATABASE_URL="postgresql://user:pass@postgres-production-7cb3.up.railway.app:PORT/railway" node scripts/verify_railway_db.js
 * 
 * Railway DATABASE_URL format:
 *   postgresql://postgres:PASSWORD@postgres-production-7cb3.up.railway.app:PORT/railway
 * (Get the full URL from Railway → Your Project → PostgreSQL service → Connect → Connection URL)
 */

const { Pool } = require('pg');
require('dotenv').config();

async function verify() {
  const connectionString = process.env.DATABASE_URL;
  if (!connectionString) {
    console.error('❌ DATABASE_URL not set. Add it to .env or pass as env var.');
    console.log('   Get it from: Railway Dashboard → PostgreSQL → Connect → Connection URL');
    process.exit(1);
  }

  // Railway: public networking (postgres-*.up.railway.app) often needs ssl: false
  const pool = new Pool({
    connectionString,
    ssl: process.env.DB_SSL !== 'false' ? { rejectUnauthorized: false } : false,
    client_encoding: 'UTF8'
  });

  try {
    console.log('🔌 Connecting to Railway PostgreSQL...');
    const client = await pool.connect();

    // 1. Table structure
    console.log('\n📋 TABLE: dictionary_entries - COLUMNS:');
    const cols = await client.query(`
      SELECT column_name, data_type, is_nullable
      FROM information_schema.columns 
      WHERE table_name = 'dictionary_entries'
      ORDER BY ordinal_position
    `);
    cols.rows.forEach(r => console.log(`   ${r.column_name}: ${r.data_type} (nullable: ${r.is_nullable})`));

    // 2. Row count
    const count = await client.query('SELECT COUNT(*) FROM dictionary_entries');
    console.log(`\n📊 Total entries: ${count.rows[0].count}`);

    // 3. Max entry_number
    const maxNum = await client.query('SELECT MAX(entry_number) as max FROM dictionary_entries');
    console.log(`   Max entry_number: ${maxNum.rows[0].max}`);

    // 4. Sample rows (first 3)
    const sample = await client.query(`
      SELECT id, entry_number, word_konkani_devanagari, word_konkani_english_alphabet, 
             english_meaning, context_usage_sentence
      FROM dictionary_entries ORDER BY entry_number LIMIT 3
    `);
    console.log('\n📝 Sample rows (first 3):');
    sample.rows.forEach((r, i) => {
      console.log(`   [${i + 1}] entry_number=${r.entry_number} | devanagari="${r.word_konkani_devanagari}" | roman="${r.word_konkani_english_alphabet}" | meaning="${r.english_meaning?.substring(0, 40)}..."`);
    });

    // 5. Check if search_vector is populated
    const svCheck = await client.query(`
      SELECT COUNT(*) as with_sv FROM dictionary_entries WHERE search_vector IS NOT NULL
    `);
    console.log(`\n🔍 search_vector populated: ${svCheck.rows[0].with_sv} / ${count.rows[0].count} rows`);

    client.release();
    console.log('\n✅ Database connection successful!');
  } catch (err) {
    console.error('❌ Connection failed:', err.message);
    process.exit(1);
  } finally {
    await pool.end();
  }
}

verify();
