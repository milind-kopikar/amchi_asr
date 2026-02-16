-- Amchigale Konkani Dictionary Database Export (Railway Compatible)
-- Generated: 2025-10-03T22:30:00.000Z
-- Entries: 4381

-- Drop table if exists to start fresh
DROP TABLE IF EXISTS dictionary_entries CASCADE;

-- Schema with auto-generating UUID
CREATE TABLE dictionary_entries (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  entry_number INTEGER UNIQUE,
  word_konkani_devanagari TEXT,
  word_konkani_english_alphabet TEXT,
  english_meaning TEXT,
  context_usage_sentence TEXT,
  devanagari_needs_correction BOOLEAN DEFAULT FALSE,
  meaning_needs_correction BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
  search_vector TSVECTOR,
  validation_status VARCHAR,
  corrected_devanagari_user_x TEXT,
  corrected_devanagari_user_y TEXT,
  corrected_meaning_user_x TEXT,
  corrected_meaning_user_y TEXT
);

-- Indexes
CREATE INDEX idx_dictionary_english_word ON dictionary_entries USING btree (word_konkani_english_alphabet);
CREATE INDEX idx_dictionary_devanagari_word ON dictionary_entries USING btree (word_konkani_devanagari);
CREATE INDEX idx_dictionary_meaning_fulltext ON dictionary_entries USING gin (to_tsvector('english'::regconfig, english_meaning));
CREATE INDEX idx_dictionary_context_fulltext ON dictionary_entries USING gin (to_tsvector('english'::regconfig, context_usage_sentence));
CREATE INDEX idx_dictionary_search_combined ON dictionary_entries USING gin (search_vector);