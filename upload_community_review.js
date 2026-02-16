/**
 * Upload community_review.csv to Railway dictionary_entries.
 * - ADD: Insert new entries
 * - UPDATE: Update existing entries (match by word_konkani_devanagari)
 *
 * Usage: node scripts/upload_community_review.js
 * Requires: DATABASE_URL in .env, DB_SSL=false for Railway public networking
 */

const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { Pool } = require('pg');
require('dotenv').config();

const CSV_FILE = path.join(__dirname, '../community_review.csv');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.DB_SSL !== 'false' ? { rejectUnauthorized: false } : false,
  client_encoding: 'UTF8'
});

function cleanText(text) {
  if (!text || typeof text !== 'string') return null;
  let cleaned = text.normalize('NFC').replace(/^\uFEFF/, '').replace(/[\u200B-\u200D\uFEFF]/g, '').trim();
  return cleaned.length > 0 ? cleaned : null;
}

async function uploadCommunityReview() {
  if (!fs.existsSync(CSV_FILE)) {
    console.error('❌ community_review.csv not found');
    process.exit(1);
  }

  const rows = [];
  await new Promise((resolve, reject) => {
    fs.createReadStream(CSV_FILE, { encoding: 'utf8' })
      .pipe(csv())
      .on('data', (row) => rows.push(row))
      .on('end', resolve)
      .on('error', reject);
  });

  const client = await pool.connect();

  try {
    const addRows = rows.filter(r => (r.Action || '').toUpperCase() === 'ADD');
    const updateRows = rows.filter(r => (r.Action || '').toUpperCase() === 'UPDATE');

    console.log(`📖 Loaded ${rows.length} rows: ${addRows.length} ADD, ${updateRows.length} UPDATE\n`);

    // Get max entry_number for new inserts
    const maxRes = await client.query('SELECT COALESCE(MAX(entry_number), 0) as max FROM dictionary_entries');
    let nextEntryNumber = parseInt(maxRes.rows[0].max, 10) + 1;

    let addCount = 0;
    let updateCount = 0;
    const errors = [];

    // Process UPDATEs first (enhance existing entries)
    for (let i = 0; i < updateRows.length; i++) {
      const r = updateRows[i];
      const devanagari = cleanText(r['Konkani Word (Devanagari)']);
      if (!devanagari) {
        errors.push({ row: r, error: 'Missing Devanagari' });
        continue;
      }

      const roman = cleanText(r['Konkani Word (Roman/IAST)']);
      const meaning = cleanText(r['English Meaning']);
      const usage = cleanText(r['Usage Example']);

      try {
        const updateResult = await client.query(
          `UPDATE dictionary_entries SET
            word_konkani_english_alphabet = COALESCE(NULLIF(TRIM($2), ''), word_konkani_english_alphabet),
            english_meaning = COALESCE(NULLIF(TRIM($3), ''), english_meaning),
            context_usage_sentence = COALESCE(NULLIF(TRIM($4), ''), context_usage_sentence),
            updated_at = CURRENT_TIMESTAMP
          WHERE word_konkani_devanagari = $1`,
          [devanagari, roman, meaning, usage]
        );

        if (updateResult.rowCount > 0) {
          updateCount++;
        } else {
          errors.push({ row: r, error: `No match for Devanagari: ${devanagari}` });
        }
      } catch (err) {
        errors.push({ row: r, error: err.message });
      }
    }

    // Update search_vector for updated rows
    const updateDevanagaris = updateRows.map(r => cleanText(r['Konkani Word (Devanagari)'])).filter(Boolean);
    if (updateDevanagaris.length > 0) {
      await client.query(`
        UPDATE dictionary_entries SET search_vector = 
          setweight(to_tsvector('simple', coalesce(word_konkani_devanagari,'')), 'A') ||
          setweight(to_tsvector('simple', coalesce(word_konkani_english_alphabet,'')), 'A') ||
          setweight(to_tsvector('english', coalesce(english_meaning,'')), 'B') ||
          setweight(to_tsvector('english', coalesce(context_usage_sentence,'')), 'C')
        WHERE word_konkani_devanagari = ANY($1::text[])
      `, [updateDevanagaris]);
    }

    console.log(`✅ Updated ${updateCount} existing entries\n`);

    // Process ADDs (new entries)
    for (let i = 0; i < addRows.length; i++) {
      const r = addRows[i];
      const devanagari = cleanText(r['Konkani Word (Devanagari)']);
      if (!devanagari) {
        errors.push({ row: r, error: 'Missing Devanagari' });
        continue;
      }

      let roman = cleanText(r['Konkani Word (Roman/IAST)']);
      if (!roman) roman = `[${devanagari}]`;
      const meaning = cleanText(r['English Meaning']);
      const usage = cleanText(r['Usage Example']);

      // Skip if already exists (avoid duplicate)
      const exists = await client.query(
        'SELECT id FROM dictionary_entries WHERE word_konkani_devanagari = $1',
        [devanagari]
      );
      if (exists.rows.length > 0) {
        errors.push({ row: r, error: `Already exists (skipped insert): ${devanagari}` });
        continue;
      }

      try {
        await client.query(
          `INSERT INTO dictionary_entries (
            entry_number, word_konkani_devanagari, word_konkani_english_alphabet,
            english_meaning, context_usage_sentence, devanagari_needs_correction,
            meaning_needs_correction, search_vector
          ) VALUES ($1, $2, $3, $4, $5, FALSE, FALSE,
            setweight(to_tsvector('simple', coalesce($2,'')), 'A') ||
            setweight(to_tsvector('simple', coalesce($3,'')), 'A') ||
            setweight(to_tsvector('english', coalesce($4,'')), 'B') ||
            setweight(to_tsvector('english', coalesce($5,'')), 'C')
          )`,
          [nextEntryNumber++, devanagari, roman, meaning, usage]
        );
        addCount++;
        if (addCount % 50 === 0) console.log(`   Inserted ${addCount}/${addRows.length}...`);
      } catch (err) {
        errors.push({ row: r, error: err.message });
      }
    }

    console.log(`\n✅ Inserted ${addCount} new entries`);
    console.log(`\n📊 Summary: ${addCount} added, ${updateCount} updated`);

    if (errors.length > 0) {
      console.log(`\n⚠️ ${errors.length} issues:`);
      errors.slice(0, 15).forEach(e => console.log(`   - ${e.error}`));
      fs.writeFileSync(path.join(__dirname, '../upload-errors.json'), JSON.stringify(errors, null, 2));
      console.log(`   Full log: upload-errors.json`);
    }
  } finally {
    client.release();
    await pool.end();
  }
}

uploadCommunityReview().catch(err => {
  console.error('❌ Fatal error:', err);
  process.exit(1);
});
