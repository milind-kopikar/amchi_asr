---
license: mit
language:
- mr
pipeline_tag: automatic-speech-recognition
library_name: nemo
---
## IndicConformer

  IndicConformer is a Hybrid CTC-RNNT conformer ASR(Automatic Speech Recognition) model.

  ### Language

  Marathi

  ### Input

  This model accepts 16000 KHz Mono-channel Audio (wav files) as input.

  ### Output

  This model provides transcribed speech as a string for a given audio sample.

  ## Model Architecture

  This model is a conformer-Large model, consisting of 120M parameters, as the encoder, with a hybrid CTC-RNNT decoder. The model has 17 conformer blocks with
  512 as the model dimension.


  ## AI4Bharat NeMo:

  To load, train, fine-tune or play with the model you will need to install [AI4Bharat NeMo](https://github.com/AI4Bharat/NeMo). We recommend you install it using the command shown below
  ```
  git clone https://github.com/AI4Bharat/NeMo.git && cd NeMo && git checkout nemo-v2 && bash reinstall.sh
  ```

  ## Usage
  Download and load the model from Huggingface.
  ```
  import torch
  import nemo.collections.asr as nemo_asr

  model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.freeze() # inference mode
  model = model.to(device) # transfer model to device
  ```
  Get an audio file ready by running the command shown below in your terminal. This will convert the audio to 16000 Hz and monochannel.
  ```
  ffmpeg -i sample_audio.wav -ac 1 -ar 16000 sample_audio_infer_ready.wav
  ```

  
  ### Inference using CTC decoder
  ```
  model.cur_decoder = "ctc"
  ctc_text = model.transcribe(['sample_audio_infer_ready.wav'], batch_size=1,logprobs=False, language_id='mr')[0]
  print(ctc_text)
  ```

  ### Inference using RNNT decoder
  ```
  model.cur_decoder = "rnnt"
  rnnt_text = model.transcribe(['sample_audio_infer_ready.wav'], batch_size=1, language_id='mr')[0]
  print(rnnt_text)
  ```
