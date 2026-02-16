# Amchigale Konkani Dictionary - Local Development Setup
# This script sets up PostgreSQL and vector database locally

## Prerequisites Installation

### 1. Install PostgreSQL (Windows)
# Download from: https://www.postgresql.org/download/windows/
# Or use chocolatey:
# choco install postgresql

### 2. Install Docker (for vector database)
# Download from: https://docs.docker.com/desktop/install/windows-install/

### 3. Install Node.js dependencies
npm install pg pg-hstore sequelize dotenv
npm install --save-dev @types/pg

## Local Database Setup Commands

### 1. Create PostgreSQL Database
# Connect to PostgreSQL as superuser
psql -U postgres

# Create database and user
CREATE DATABASE konkani_dictionary;
CREATE USER konkani_dev WITH PASSWORD 'dev_password_2024';
GRANT ALL PRIVILEGES ON DATABASE konkani_dictionary TO konkani_dev;

# Exit psql
\q

### 2. Run Schema Creation
# Execute the schema file
psql -U konkani_dev -d konkani_dictionary -f database/schema.sql

### 3. Setup Vector Database (Qdrant)
# Pull and run Qdrant vector database
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant

## Environment Configuration

Create .env file with:
```
# PostgreSQL Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=konkani_dictionary
DB_USER=konkani_dev
DB_PASSWORD=dev_password_2024

# Vector Database Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here

# Embedding Model Configuration
OPENAI_API_KEY=your_openai_key_for_embeddings
EMBEDDING_MODEL=text-embedding-3-small

# Application Configuration
NODE_ENV=development
PORT=3001
```

## Google Cloud Migration Path

### 1. PostgreSQL on Google Cloud
- Use Cloud SQL for PostgreSQL
- Managed service with automatic backups
- Easy scaling and high availability

### 2. Vector Database Options
- Vertex AI Vector Search (Google's managed solution)
- Self-hosted Qdrant on GKE
- Pinecone (third-party managed)

### 3. Migration Commands (Future)
```bash
# Export local data
pg_dump -U konkani_dev konkani_dictionary > backup.sql

# Import to Cloud SQL
gcloud sql import sql INSTANCE_NAME gs://BUCKET_NAME/backup.sql --database=konkani_dictionary

# Vector data migration
# Custom script to re-embed and upload to cloud vector DB
```

## Next Steps
1. Run the setup commands above
2. Install the npm packages
3. Create database models in Node.js
4. Implement spreadsheet import functionality
5. Create vector embeddings pipeline