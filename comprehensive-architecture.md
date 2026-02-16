# Konkani Language Platform - Complete Database Architecture
# Supports Dictionary, ASR Training, WhatsApp Validation, Multi-Geography

## 🏗️ Database Architecture Overview

### 1. PRIMARY DATABASES

#### PostgreSQL (Main Operational Database)
- **Dictionary entries & validation workflow**
- **Expert management & geography mapping**  
- **ASR training data metadata**
- **WhatsApp integration logs**
- **User contributions & reputation system**

#### Vector Database (Qdrant/Pinecone)
- **Semantic search for dictionary entries**
- **Language similarity matching across dialects**
- **RAG context for LLM interactions**
- **Phonetic similarity vectors (for ASR)**

#### Audio File Storage (Google Cloud Storage/AWS S3)
- **Raw audio recordings from WhatsApp experts**
- **Processed audio files for ASR training**
- **Backup audio samples**
- **Synthetic audio generations**

#### Redis (Caching & Queue Management)
- **WhatsApp message queuing**
- **Real-time validation status**
- **Expert assignment queue**
- **ASR training job queue**

### 2. SPECIALIZED STORAGE

#### Time-Series Database (InfluxDB/TimescaleDB)
- **ASR model performance metrics over time**
- **Expert validation response times**
- **Geographic usage patterns**
- **Model accuracy tracking by region**

#### Graph Database (Neo4j) - Optional for Advanced Features
- **Dialect relationships between regions**
- **Expert expertise networks**
- **Word etymology and linguistic connections**
- **Social validation patterns**

## 🗄️ Core PostgreSQL Schema Design

### Geography & Dialect Management
```sql
-- Geographic regions and their dialect characteristics
CREATE TABLE geographic_regions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    region_name VARCHAR(100) NOT NULL,  -- "North Goa", "South Karnataka", etc.
    country VARCHAR(50) DEFAULT 'India',
    state_province VARCHAR(100),
    district VARCHAR(100),
    
    -- Linguistic characteristics
    dialect_code VARCHAR(20) UNIQUE,     -- "NG_KON", "SK_KON", etc.
    primary_script VARCHAR(20),          -- "Devanagari", "Roman", "Kannada"
    linguistic_notes TEXT,
    
    -- Demographics
    estimated_speakers INTEGER,
    speaker_density VARCHAR(20),         -- "high", "medium", "low"
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Expert contributors mapped to their regions
CREATE TABLE expert_contributors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Personal info
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(20),
    whatsapp_number VARCHAR(20) UNIQUE,
    
    -- Geographic expertise
    primary_region_id UUID REFERENCES geographic_regions(id),
    secondary_regions UUID[], -- Array of region IDs for multi-region experts
    
    -- Expertise level
    expertise_level VARCHAR(20) DEFAULT 'contributor', -- contributor, expert, senior_expert, regional_authority
    specialization TEXT[], -- ['grammar', 'vocabulary', 'pronunciation', 'literature']
    
    -- Validation authority
    can_validate_others BOOLEAN DEFAULT FALSE,
    validation_tier INTEGER DEFAULT 1,   -- 1=basic contributor, 2=expert validator, 3=final authority
    
    -- Performance metrics  
    contributions_count INTEGER DEFAULT 0,
    validation_accuracy DECIMAL(5,2) DEFAULT 0.0,
    response_time_avg_hours DECIMAL(8,2),
    reputation_score INTEGER DEFAULT 100,
    
    -- WhatsApp integration
    whatsapp_verified BOOLEAN DEFAULT FALSE,
    whatsapp_bot_session_id VARCHAR(100),
    preferred_language VARCHAR(10) DEFAULT 'en', -- for bot communication
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP
);
```

### Enhanced Dictionary Schema
```sql
-- Main dictionary with geographic variations
CREATE TABLE dictionary_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Core word data (from your spreadsheet)
    entry_number INTEGER,
    word_konkani_devanagari TEXT,
    word_konkani_english_alphabet TEXT NOT NULL,
    english_meaning TEXT NOT NULL,
    context_usage_sentence TEXT,
    
    -- Geographic context
    primary_region_id UUID REFERENCES geographic_regions(id),
    regional_variations JSONB, -- {"NG_KON": {"spelling": "...", "pronunciation": "..."}}
    
    -- Original validation flags
    devanagari_needs_correction BOOLEAN DEFAULT FALSE,
    meaning_needs_correction BOOLEAN DEFAULT FALSE,
    
    -- Enhanced validation workflow
    validation_status VARCHAR(20) DEFAULT 'pending', -- pending, expert_review, validated, disputed, archived
    validation_tier_reached INTEGER DEFAULT 0,        -- 0=unvalidated, 1=expert_approved, 2=authority_approved
    validation_confidence DECIMAL(5,2) DEFAULT 0.0,  -- 0-100 based on expert consensus
    
    -- Audio integration for ASR
    has_audio_samples BOOLEAN DEFAULT FALSE,
    audio_samples_count INTEGER DEFAULT 0,
    phonetic_transcription TEXT, -- IPA or custom phonetic notation
    
    -- Linguistic metadata
    part_of_speech VARCHAR(50),
    word_frequency VARCHAR(20),   -- "common", "uncommon", "rare", "archaic"
    register VARCHAR(20),         -- "formal", "informal", "literary", "colloquial"
    
    -- Version control
    version INTEGER DEFAULT 1,
    parent_entry_id UUID,        -- For tracking word evolution
    
    -- System fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validated_at TIMESTAMP,
    search_vector tsvector
);

-- WhatsApp validation workflow
CREATE TABLE validation_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- What's being validated
    entry_id UUID REFERENCES dictionary_entries(id),
    validation_type VARCHAR(30) NOT NULL, -- "spelling", "meaning", "pronunciation", "new_word"
    
    -- Original vs suggested values
    original_value TEXT,
    suggested_value TEXT,
    suggestion_reason TEXT,
    
    -- Geographic context
    region_id UUID REFERENCES geographic_regions(id),
    dialect_specific BOOLEAN DEFAULT FALSE,
    
    -- WhatsApp workflow
    whatsapp_message_id VARCHAR(100),
    expert_id UUID REFERENCES expert_contributors(id),
    assigned_at TIMESTAMP,
    responded_at TIMESTAMP,
    
    -- Expert response
    expert_response VARCHAR(20), -- "approved", "rejected", "modified", "needs_review"
    expert_comments TEXT,
    expert_suggested_value TEXT,
    confidence_level INTEGER, -- 1-5 scale
    
    -- Escalation workflow
    escalation_required BOOLEAN DEFAULT FALSE,
    escalated_to UUID REFERENCES expert_contributors(id),
    escalation_reason TEXT,
    
    -- Final resolution
    status VARCHAR(20) DEFAULT 'pending', -- pending, approved, rejected, escalated, resolved
    resolved_by UUID REFERENCES expert_contributors(id),
    resolution_notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### ASR Training Data Schema
```sql
-- Audio recordings for ASR training
CREATE TABLE asr_audio_recordings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Associated text
    entry_id UUID REFERENCES dictionary_entries(id),
    text_content TEXT NOT NULL,
    text_normalized TEXT, -- Preprocessing for training
    
    -- Audio file details
    audio_file_path VARCHAR(500) NOT NULL,
    audio_file_size_bytes BIGINT,
    duration_seconds DECIMAL(8,3),
    sample_rate INTEGER DEFAULT 16000,
    audio_format VARCHAR(10) DEFAULT 'wav',
    
    -- Speaker information
    speaker_id UUID REFERENCES expert_contributors(id),
    speaker_gender VARCHAR(10),
    speaker_age_range VARCHAR(20), -- "20-30", "30-40", etc.
    
    -- Geographic context
    region_id UUID REFERENCES geographic_regions(id),
    recording_location VARCHAR(100),
    dialect_code VARCHAR(20),
    
    -- Recording quality
    audio_quality_score DECIMAL(3,2), -- 1.0-5.0 scale
    background_noise_level VARCHAR(20), -- "low", "medium", "high"
    recording_environment VARCHAR(50), -- "studio", "home", "outdoor", etc.
    
    -- Training usage
    dataset_split VARCHAR(10) DEFAULT 'train', -- train, validation, test
    used_in_training BOOLEAN DEFAULT FALSE,
    training_weight DECIMAL(3,2) DEFAULT 1.0,
    
    -- WhatsApp integration
    whatsapp_message_id VARCHAR(100),
    received_via_whatsapp BOOLEAN DEFAULT TRUE,
    
    -- Processing status
    preprocessing_status VARCHAR(20) DEFAULT 'pending', -- pending, processed, failed
    transcription_verified BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

-- ASR model training metadata
CREATE TABLE asr_training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Model details
    base_model_name VARCHAR(100), -- "ai4bharat/indicconformer_hi" (Marathi base)
    model_version VARCHAR(50),
    training_framework VARCHAR(50) DEFAULT 'NeMo',
    
    -- Training data
    total_audio_hours DECIMAL(8,2),
    training_samples_count INTEGER,
    validation_samples_count INTEGER,
    test_samples_count INTEGER,
    
    -- Geographic focus
    target_regions UUID[], -- Array of region IDs
    dialect_weights JSONB, -- {"NG_KON": 0.4, "SK_KON": 0.6}
    
    -- Training parameters
    training_config JSONB, -- Store complete NeMo config
    epochs_completed INTEGER DEFAULT 0,
    learning_rate DECIMAL(10,6),
    batch_size INTEGER,
    
    -- Performance metrics
    final_wer DECIMAL(5,3), -- Word Error Rate
    final_cer DECIMAL(5,3), -- Character Error Rate
    validation_loss DECIMAL(10,6),
    
    -- Regional performance breakdown
    regional_performance JSONB, -- {"NG_KON": {"wer": 0.15, "cer": 0.08}}
    
    -- Training status
    status VARCHAR(20) DEFAULT 'pending', -- pending, training, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Model artifacts
    model_checkpoint_path VARCHAR(500),
    tensorboard_logs_path VARCHAR(500),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔄 Workflow Integration Architecture

### WhatsApp Validation Flow
```
1. New word/correction needed
2. System assigns to regional expert via WhatsApp
3. Expert responds with validation
4. If disputed → escalates to senior expert
5. Final validation updates main dictionary
6. Regional variations tracked separately
```

### ASR Training Pipeline
```
1. Expert receives text via WhatsApp
2. Records audio pronunciation  
3. Audio uploaded to cloud storage
4. Metadata stored in PostgreSQL
5. Batch processing for training data prep
6. NeMo training with regional weighting
7. Model evaluation across all regions
```

## 🌐 Google Cloud Deployment Strategy

### Recommended Services
- **Cloud SQL (PostgreSQL)** - Main database
- **Vertex AI Vector Search** - Semantic search
- **Cloud Storage** - Audio files
- **Cloud Run** - WhatsApp bot & API services  
- **Vertex AI** - Custom ASR model training
- **Pub/Sub** - Message queuing
- **Cloud Functions** - Event processing

### Data Flow
```
WhatsApp ↔ Cloud Run (Bot) ↔ Pub/Sub ↔ Cloud Functions ↔ Cloud SQL
                ↓                                           ↓
        Cloud Storage (Audio) ← → Vertex AI (ASR Training)
                ↓                                           ↓
        Vector Search ← → Your Web App ← → Users
```

## 📈 Scalability Considerations

### Multi-Language Extension
- Same schema supports multiple languages
- Language-specific expert pools
- Cross-linguistic similarity analysis
- Shared ASR training techniques

### Regional Scaling
- Partition data by geographic region
- Regional model specialization
- Local expert communities
- Distributed validation workflows

This architecture supports your complete vision from simple dictionary to advanced ASR while maintaining data integrity and expert validation workflows. Ready to implement step by step!