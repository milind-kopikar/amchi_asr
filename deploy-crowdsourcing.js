// Deploy crowdsourcing schema to Railway database
const { Pool } = require('pg');
require('dotenv').config();

async function deployCrowdsourcingSchema() {
  // Use Railway production environment variables directly
  const pool = new Pool({
    host: 'postgres.railway.internal',
    port: 5432,
    database: 'railway',
    user: 'postgres',
    password: 'cCIxhdPypKMWnUciqcZvZqJRtQLblAve',
    ssl: { 
      rejectUnauthorized: false,
      require: true 
    }
  });

  try {
    console.log('🚀 Deploying crowdsourcing schema to Railway...');

    // Create tables
    await pool.query(`
      -- Users/Contributors table
      CREATE TABLE IF NOT EXISTS contributors (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        email VARCHAR(255) UNIQUE,
        name VARCHAR(100),
        is_expert BOOLEAN DEFAULT FALSE,
        is_active BOOLEAN DEFAULT TRUE,
        contributions_count INTEGER DEFAULT 0,
        approved_contributions INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT NOW(),
        last_login TIMESTAMP,
        password_hash VARCHAR(255)
      );
    `);
    console.log('✅ Contributors table created');

    await pool.query(`
      -- Suggested changes/additions
      CREATE TABLE IF NOT EXISTS dictionary_suggestions (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        original_entry_id UUID REFERENCES dictionary_entries(id),
        contributor_id UUID REFERENCES contributors(id),
        suggestion_type VARCHAR(20) CHECK (suggestion_type IN ('correction', 'addition', 'deletion')),
        
        -- Original values (for corrections)
        original_word_konkani_devanagari TEXT,
        original_word_konkani_english_alphabet TEXT,
        original_english_meaning TEXT,
        original_context_usage_sentence TEXT,
        
        -- Suggested values
        suggested_word_konkani_devanagari TEXT,
        suggested_word_konkani_english_alphabet TEXT,
        suggested_english_meaning TEXT,
        suggested_context_usage_sentence TEXT,
        
        -- Metadata
        contributor_notes TEXT,
        status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'in_review')),
        
        -- Review information
        reviewed_by UUID REFERENCES contributors(id),
        reviewed_at TIMESTAMP,
        reviewer_notes TEXT,
        
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      );
    `);
    console.log('✅ Dictionary suggestions table created');

    await pool.query(`
      -- Suggestion votes/ratings
      CREATE TABLE IF NOT EXISTS suggestion_votes (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        suggestion_id UUID REFERENCES dictionary_suggestions(id),
        contributor_id UUID REFERENCES contributors(id),
        vote_type VARCHAR(10) CHECK (vote_type IN ('helpful', 'not_helpful')),
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(suggestion_id, contributor_id)
      );
    `);
    console.log('✅ Suggestion votes table created');

    await pool.query(`
      -- Change history/audit log
      CREATE TABLE IF NOT EXISTS dictionary_change_log (
        id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
        entry_id UUID REFERENCES dictionary_entries(id),
        suggestion_id UUID REFERENCES dictionary_suggestions(id),
        change_type VARCHAR(20),
        old_values JSONB,
        new_values JSONB,
        changed_by UUID REFERENCES contributors(id),
        approved_by UUID REFERENCES contributors(id),
        created_at TIMESTAMP DEFAULT NOW()
      );
    `);
    console.log('✅ Change log table created');

    // Create indexes
    await pool.query(`
      CREATE INDEX IF NOT EXISTS idx_suggestions_status ON dictionary_suggestions(status);
      CREATE INDEX IF NOT EXISTS idx_suggestions_contributor ON dictionary_suggestions(contributor_id);
      CREATE INDEX IF NOT EXISTS idx_suggestions_entry ON dictionary_suggestions(original_entry_id);
      CREATE INDEX IF NOT EXISTS idx_suggestions_created ON dictionary_suggestions(created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_contributors_expert ON contributors(is_expert);
      CREATE INDEX IF NOT EXISTS idx_change_log_entry ON dictionary_change_log(entry_id);
    `);
    console.log('✅ Indexes created');

    // Insert sample expert users
    await pool.query(`
      INSERT INTO contributors (email, name, is_expert, is_active, password_hash) VALUES 
      ('expert@konkani.org', 'Dr. Konkani Expert', TRUE, TRUE, '$2b$10$placeholder_hash_for_password'),
      ('admin@konkani.org', 'Dictionary Admin', TRUE, TRUE, '$2b$10$placeholder_hash_for_password')
      ON CONFLICT (email) DO NOTHING;
    `);
    console.log('✅ Sample expert users created');

    console.log('🎉 Crowdsourcing schema deployed successfully!');
    console.log('');
    console.log('📋 Next steps:');
    console.log('1. Update expert passwords in the database');
    console.log('2. Deploy updated server.js to Railway');
    console.log('3. Test the suggestion and review system');
    
  } catch (error) {
    console.error('❌ Error deploying schema:', error);
  } finally {
    await pool.end();
  }
}

deployCrowdsourcingSchema();