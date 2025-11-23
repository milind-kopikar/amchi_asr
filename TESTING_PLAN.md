# Testing Konkani ASR with 2-3 Simple Sentences

## 🎯 **Plan: Start Small, Scale Big**

**Perfect approach!** Let's test with 2-3 simple Konkani sentences first, then scale up to your 10-minute audio.

## 📋 **Step-by-Step Testing Plan**

### **Phase 1: Record 2-3 Simple Sentences** (5-10 minutes)

#### **Step 1.1: Prepare Recording Scripts**
```bash
cd c:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr

# Create recording setup for 3 short sentences
python scripts/prepare_recording.py --audio_name test_sentences.wav --duration 30
```

This creates:
- `record_test_sentences.ps1` - PowerShell recording script
- `test_sentences_transcript.txt` - Template for your transcript

#### **Step 1.2: Record Your Sentences**
1. **Open PowerShell** in the project directory
2. **Run the recording script**:
   ```powershell
   .\record_test_sentences.ps1
   ```
3. **Speak 2-3 simple Konkani sentences**, for example:
   - "माझे नाव मिलिंद आहे." (My name is Milind.)
   - "आमी कोकणात राहतो." (We live in Konkan.)
   - "कोकणी भाषा छान आहे." (Konkani language is nice.)

#### **Step 1.3: Create Transcript File**
Edit `test_sentences_transcript.txt` with the exact words you spoke:

```
माझे नाव मिलिंद आहे. आमी कोकणात राहतो. कोकणी भाषा छान आहे.
```

### **Phase 2: Test Block 1 (Data Management)** ✅

#### **Step 2.1: Prepare Data**
```bash
# Create data directory structure
python scripts/setup_environment.py

# Prepare your test data
python scripts/prepare_data.py \
  --audio_dir . \
  --transcript_dir . \
  --output_dir data/test_run
```

#### **Step 2.2: Verify Manifest Creation**
Check that these files were created:
- `data/test_run/train.tsv`
- `data/test_run/val.tsv`
- `data/test_run/test.tsv`

### **Phase 3: Test Block 2 (Model Management)** ✅

#### **Step 3.1: Download Base Model**
```bash
# Download IndicConformer model
python scripts/download_model.py
```

This downloads the Marathi ASR model that we'll fine-tune for Konkani.

### **Phase 4: Test Block 3 (Training Pipeline)** ✅

#### **Step 4.1: Quick Training Test**
```bash
# Run minimal training test (fast, just to verify pipeline)
python scripts/minimal_test.py \
  --audio_file test_sentences.wav \
  --transcript "माझे नाव मिलिंद आहे. आमी कोकणात राहतो. कोकणी भाषा छान आहे."
```

This tests:
- Audio loading and preprocessing
- Model loading
- Basic inference (transcription)
- End-to-end pipeline

### **Phase 5: Test Block 4 (Evaluation)** ✅

#### **Step 5.1: Evaluate Results**
```bash
# Run evaluation on your test data
python scripts/evaluate.py \
  --model_path models/indicconformer_marathi.nemo \
  --manifest data/test_run/test.tsv \
  --output_file evaluation_results.json
```

### **Phase 6: Test Block 5 (Inference)** ✅

#### **Step 6.1: Test Transcription**
```bash
# Transcribe your recorded audio
python scripts/infer.py \
  --model_path models/indicconformer_marathi.nemo \
  --audio_file test_sentences.wav \
  --output_file transcription_results.json
```

## 🎯 **Expected Results**

### **What Should Happen:**
1. **Block 1**: Manifest files created successfully
2. **Block 2**: Model downloads (takes 5-10 minutes)
3. **Block 3**: Minimal test runs and shows transcription attempt
4. **Block 4**: Evaluation metrics calculated
5. **Block 5**: Your audio gets transcribed (in Marathi, not perfect Konkani yet)

### **Success Criteria:**
- ✅ No crashes/errors in any step
- ✅ Manifest files contain your data
- ✅ Model loads successfully
- ✅ Audio gets processed without errors
- ✅ You get transcription output

## 🚀 **Scaling Up Plan**

### **After Successful Test Run:**

#### **Phase 1: Segment Your 10-Minute Audio**
```bash
# Split your 10-minute story into 30-second segments
# Create corresponding transcripts for each segment
```

#### **Phase 2: Full Training**
```bash
# Run complete fine-tuning with your segmented data
python scripts/fine_tune.py --config configs/konkani_finetune.yaml
```

#### **Phase 3: Evaluate & Iterate**
```bash
# Test the fine-tuned model
python scripts/infer.py --model_path results/checkpoints/konkani_model.nemo --audio_file your_test_audio.wav
```

## 📊 **Timeline**

| Phase | Duration | What Happens |
|-------|----------|--------------|
| **Recording** | 5-10 min | Record 2-3 sentences |
| **Block 1-2** | 15-20 min | Setup data & download model |
| **Block 3-5** | 10-15 min | Test end-to-end pipeline |
| **Analysis** | 5 min | Review results & plan next steps |
| **Total Test** | **35-50 min** | Complete pipeline validation |

## 🛠️ **Troubleshooting**

### **If Recording Fails:**
```bash
# Check microphone setup
python scripts/setup_environment.py
```

### **If Model Download Fails:**
```bash
# Try manual download
pip install huggingface_hub
huggingface-cli download ai4bharat/indicconformer_marathi --local-dir models/
```

### **If Any Step Fails:**
- Check the error message
- Verify file paths
- Ensure all dependencies are installed
- Try running individual components

## 🎉 **Success = Ready to Scale!**

Once this test run works, you'll know:
- ✅ Your environment is set up correctly
- ✅ The pipeline works end-to-end
- ✅ You can process Konkani audio
- ✅ Ready to add more data and train properly

**Then we can confidently scale up to your 10-minute audio and beyond!**

---

**Ready to start? Let's record those 2-3 sentences and test the pipeline! 🚀**</content>
<parameter name="filePath">C:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\TESTING_PLAN.md