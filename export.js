const { Pool } = require('pg');
const fs = require('fs');
require('dotenv').config();

/**
 * Export database for Railway migration
 * Creates both schema and data files for easy deployment
 */

const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'konkani_dictionary',
  user: process.env.DB_USER || 'konkani_dev',
  password: process.env.DB_PASSWORD,
  client_encoding: 'UTF8'
});

async function exportDatabase() {
  try {
    console.log('🔄 Exporting Konkani Dictionary database...');
    
    // Export schema
    const schemaQuery = `
      SELECT 
        'CREATE TABLE IF NOT EXISTS dictionary_entries (' ||
        string_agg(
          column_name || ' ' || 
          CASE 
            WHEN data_type = 'character varying' THEN 'VARCHAR'
            WHEN data_type = 'text' THEN 'TEXT'
            WHEN data_type = 'integer' THEN 'INTEGER'
            WHEN data_type = 'boolean' THEN 'BOOLEAN'
            WHEN data_type = 'uuid' THEN 'UUID'
            WHEN data_type = 'timestamp with time zone' THEN 'TIMESTAMP WITH TIME ZONE'
            ELSE UPPER(data_type)
          END ||
          CASE 
            WHEN is_nullable = 'NO' THEN ' NOT NULL'
            ELSE ''
          END,
          ', '
        ) || ');' as create_statement
      FROM information_schema.columns 
      WHERE table_name = 'dictionary_entries'
      GROUP BY table_name;
    `;
    
    const schemaResult = await pool.query(schemaQuery);
    
    // Get indexes
    const indexQuery = `
      SELECT indexdef 
      FROM pg_indexes 
      WHERE tablename = 'dictionary_entries' AND indexname != 'dictionary_entries_pkey';
    `;
    
    const indexResult = await pool.query(indexQuery);
    
    // Export data in batches
    console.log('📊 Exporting data...');
    const dataQuery = 'SELECT * FROM dictionary_entries ORDER BY entry_number';
    const dataResult = await pool.query(dataQuery);
    
    console.log(`✅ Found ${dataResult.rows.length} entries to export`);
    
    // Create SQL file
    let sqlContent = `-- Amchigale Konkani Dictionary Database Export
-- Generated: ${new Date().toISOString()}
-- Entries: ${dataResult.rows.length}

-- Schema
${schemaResult.rows[0]?.create_statement || '-- No schema found'}

-- Indexes
${indexResult.rows.map(row => row.indexdef + ';').join('\n')}

-- Data
`;
    
    // Add data in chunks
    const chunkSize = 100;
    for (let i = 0; i < dataResult.rows.length; i += chunkSize) {
      const chunk = dataResult.rows.slice(i, i + chunkSize);
      
      sqlContent += `\n-- Batch ${Math.floor(i/chunkSize) + 1}\n`;
      sqlContent += `INSERT INTO dictionary_entries (
        entry_number, word_konkani_devanagari, word_konkani_english_alphabet, 
        english_meaning, context_usage_sentence, devanagari_needs_correction, 
        meaning_needs_correction, corrected_devanagari_user_x, corrected_devanagari_user_y,
        corrected_meaning_user_x, corrected_meaning_user_y
      ) VALUES\n`;
      
      const values = chunk.map(row => {
        const escapeValue = (val) => {
          if (val === null || val === undefined) return 'NULL';
          if (typeof val === 'boolean') return val ? 'TRUE' : 'FALSE';
          if (typeof val === 'number') return val.toString();
          return `'${val.toString().replace(/'/g, "''")}'`;
        };
        
        return `(${escapeValue(row.entry_number)}, ${escapeValue(row.word_konkani_devanagari)}, ${escapeValue(row.word_konkani_english_alphabet)}, ${escapeValue(row.english_meaning)}, ${escapeValue(row.context_usage_sentence)}, ${escapeValue(row.devanagari_needs_correction)}, ${escapeValue(row.meaning_needs_correction)}, ${escapeValue(row.corrected_devanagari_user_x)}, ${escapeValue(row.corrected_devanagari_user_y)}, ${escapeValue(row.corrected_meaning_user_x)}, ${escapeValue(row.corrected_meaning_user_y)})`;
      });
      
      sqlContent += values.join(',\n') + '\nON CONFLICT (entry_number) DO NOTHING;\n';
    }
    
    // Write to file
    const filename = 'database/konkani_dictionary_export.sql';
    fs.writeFileSync(filename, sqlContent);
    
    // Also create a JSON export for easier handling
    const jsonFilename = 'database/konkani_dictionary_export.json';
    fs.writeFileSync(jsonFilename, JSON.stringify(dataResult.rows, null, 2));
    
    console.log(`✅ Database exported successfully!`);
    console.log(`📄 SQL file: ${filename} (${(fs.statSync(filename).size / 1024 / 1024).toFixed(2)} MB)`);
    console.log(`📄 JSON file: ${jsonFilename} (${(fs.statSync(jsonFilename).size / 1024 / 1024).toFixed(2)} MB)`);
    
    // Create railway migration script
    const migrationScript = `#!/usr/bin/env node
const { Pool } = require('pg');
const fs = require('fs');

async function migrateToRailway() {
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL
  });
  
  try {
    console.log('🚀 Migrating to Railway PostgreSQL...');
    
    const sqlContent = fs.readFileSync('./database/konkani_dictionary_export.sql', 'utf8');
    await pool.query(sqlContent);
    
    console.log('✅ Migration completed!');
  } catch (error) {
    console.error('❌ Migration failed:', error);
  } finally {
    await pool.end();
  }
}

migrateToRailway();
`;
    
    fs.writeFileSync('database/migrate-to-railway.js', migrationScript);
    console.log(`📝 Railway migration script created: database/migrate-to-railway.js`);
    
  } catch (error) {
    console.error('❌ Export failed:', error);
  } finally {
    await pool.end();
  }
}

exportDatabase();