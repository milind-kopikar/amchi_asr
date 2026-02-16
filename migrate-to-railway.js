#!/usr/bin/env node
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
