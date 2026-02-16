# 📚 Amchigale Konkani Dictionary - Complete Setup Guide

## 🎯 Overview
This setup creates a dual-database system for your Konkani dictionary:
- **PostgreSQL**: Structured data matching your spreadsheet exactly
- **Vector Database**: For RAG and semantic search capabilities

## 📊 Database Schema Summary

### Main Table: `dictionary_entries`
Maps directly to your spreadsheet columns:

| Database Column | Your Spreadsheet Column |
|---|---|
| `entry_number` | Number |
| `word_konkani_devanagari` | Konkani Word in Devnagiri |
| `word_konkani_english_alphabet` | Konkani word in English alphabet |
| `english_meaning` | English Meaning |
| `context_usage_sentence` | Context/usage in a sentence |
| `devanagari_needs_correction` | Does the Devnagiri Spelling need correction? (Y/N) |
| `meaning_needs_correction` | Does the meaning need correction? (Y/N) |
| `corrected_devanagari_user_x` | Corrected Devnagiri Spelling by user X |
| `corrected_devanagari_user_y` | Corrected Devnagiri Spelling by user Y |
| `corrected_meaning_user_x` | Corrected English meaning by user X |
| `corrected_meaning_user_y` | Corrected English meaning by user Y |

### Additional Tables:
- `user_corrections`: Tracks new corrections and voting
- `new_word_submissions`: Handles crowdsourced new words
- `users`: User management for the platform

## 🚀 Local Setup Steps

### 1. Install PostgreSQL
```bash
# Download from: https://www.postgresql.org/download/windows/
# During installation, set a password for 'postgres' user
```

### 2. Create Database
```sql
-- Connect to PostgreSQL as admin
psql -U postgres

-- Create database and user
CREATE DATABASE konkani_dictionary;
CREATE USER konkani_dev WITH PASSWORD 'dev_password_2024';
GRANT ALL PRIVILEGES ON DATABASE konkani_dictionary TO konkani_dev;

-- Exit
\q
```

### 3. Setup Database Schema
```bash
cd "C:\Users\Milind Kopikare\Code\chatbot-backend"
psql -U konkani_dev -d konkani_dictionary -f database/schema.sql
```

### 4. Setup Vector Database (Qdrant)
```bash
# Install Docker Desktop first: https://docs.docker.com/desktop/install/windows-install/

# Run Qdrant
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

## 📥 Import Your Spreadsheet Data

### 1. Export Your Spreadsheet to CSV
- Open your spreadsheet
- File → Export → CSV
- Save as `konkani-dictionary.csv`

### 2. Import to Database
```bash
cd "C:\Users\Milind Kopikare\Code\chatbot-backend"
node database/import-spreadsheet.js path/to/your/konkani-dictionary.csv
```

## 🔍 Verify Setup

### Check PostgreSQL Data
```sql
-- Connect to database
psql -U konkani_dev -d konkani_dictionary

-- Check imported data
SELECT COUNT(*) FROM dictionary_entries;
SELECT * FROM dictionary_entries LIMIT 5;

-- Check entries needing correction
SELECT entry_number, word_konkani_english_alphabet, english_meaning 
FROM dictionary_entries 
WHERE devanagari_needs_correction = true OR meaning_needs_correction = true;
```

### Check Vector Database
```bash
# Check if Qdrant is running
curl http://localhost:6333/collections
```

## 🌐 Google Cloud Migration (Future)

### PostgreSQL → Cloud SQL
```bash
# Export local data
pg_dump -U konkani_dev konkani_dictionary > konkani-backup.sql

# Import to Cloud SQL (when ready)
gcloud sql import sql INSTANCE_NAME gs://BUCKET_NAME/konkani-backup.sql --database=konkani_dictionary
```

### Vector Database → Vertex AI
- Use Google's Vertex AI Vector Search
- Or deploy Qdrant to Google Kubernetes Engine

## 🤖 Next Steps: RAG Integration

1. **Generate Embeddings**: Create vector embeddings for all dictionary entries
2. **LLM Integration**: Connect Gemini/OpenAI to query the dictionary
3. **Semantic Search**: Enable meaning-based word lookup
4. **Crowdsourcing UI**: Build interface for user corrections

## 📋 Quick Commands Reference

```bash
# Start PostgreSQL service (Windows)
net start postgresql-x64-14

# Connect to database
psql -U konkani_dev -d konkani_dictionary

# Start vector database
docker start qdrant_container_name

# Import spreadsheet
node database/import-spreadsheet.js your-file.csv

# Check database status
psql -U konkani_dev -d konkani_dictionary -c "SELECT COUNT(*) FROM dictionary_entries;"
```

## 🎯 Architecture Benefits

✅ **Exact Match**: Database structure matches your spreadsheet columns  
✅ **Crowdsourcing Ready**: Built-in correction and voting system  
✅ **Multilingual Support**: Handles Devanagari and Roman scripts  
✅ **Cloud Ready**: Easy migration to Google Cloud  
✅ **RAG Enabled**: Vector search for AI integration  
✅ **Scalable**: Supports thousands of users and entries  

Ready to proceed with the setup! 🚀