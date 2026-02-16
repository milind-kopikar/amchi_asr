const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

async function migrate() {
  const client = new Client({
    host: process.env.PGHOST,
    port: process.env.PGPORT,
    database: process.env.PGDATABASE,
    user: process.env.PGUSER,
    password: process.env.PGPASSWORD,
    ssl: { rejectUnauthorized: false }
  });

  try {
    console.log('🔌 Connecting to PostgreSQL...');
    await client.connect();
    console.log('✅ Connected to database');

    console.log('📖 Reading SQL file...');
    const sqlPath = path.join(__dirname, 'database', 'konkani_dictionary_export.sql');
    const sql = fs.readFileSync(sqlPath, 'utf8');
    console.log(`📄 SQL file loaded (${sql.length} characters)`);

    console.log('🚀 Executing SQL import...');
    await client.query(sql);
    console.log('✅ Database import completed successfully');

    console.log('📊 Verifying import...');
    const result = await client.query('SELECT COUNT(*) FROM dictionary_entries');
    console.log(`✅ Import verified: ${result.rows[0].count} entries imported`);

  } catch (error) {
    console.error('❌ Migration failed:', error.message);
    process.exit(1);
  } finally {
    await client.end();
    console.log('🔌 Database connection closed');
  }
}

migrate();