import os
import tarfile
import tempfile
import yaml
import glob
import sentencepiece as spm


def test_nemo_and_local_tokenizer_match():
    # Load model path from config
    cfg_path = 'configs/konkani_finetune.yaml'
    if not os.path.exists(cfg_path):
        raise AssertionError(f'Config not found: {cfg_path}')
    with open(cfg_path, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    nemo_path = cfg.get('model', {}).get('nemo_model')
    assert nemo_path, 'nemo_model not set in config'
    assert os.path.exists(nemo_path), f'.nemo file not found: {nemo_path}'

    # Extract model_config.yaml from .nemo
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(nemo_path, 'r') as tar:
            members = {m.name: m for m in tar.getmembers()}
            assert 'model_config.yaml' in members, 'model_config.yaml not in .nemo'
            tar.extract('model_config.yaml', path=td)
            mc_path = os.path.join(td, 'model_config.yaml')

            with open(mc_path, 'r', encoding='utf-8') as f:
                mcfg = yaml.safe_load(f)

            # Read tokenizer spe_tokenizer_vocab reference if present
            spe_vocab_ref = mcfg.get('tokenizer', {}).get('spe_tokenizer_vocab')
            tokenizer_type = mcfg.get('tokenizer', {}).get('type')

            # If a spe_tokenizer_vocab reference exists, extract and count lines
            if spe_vocab_ref and isinstance(spe_vocab_ref, str) and spe_vocab_ref.startswith('nemo:'):
                fname = spe_vocab_ref.split(':', 1)[1]
                # find member matching *fname
                found = None
                for name in members.keys():
                    if name.endswith(fname):
                        found = name
                        break
                assert found, f'{fname} not found in .nemo'
                tar.extract(found, path=td)
                extracted_vocab = os.path.join(td, found)
                # count vocab lines
                with open(extracted_vocab, 'r', encoding='utf-8') as vf:
                    lines = [l.rstrip('\n') for l in vf if l.strip()]
                nemo_vocab_size = len(lines)
            else:
                nemo_vocab_size = None

    # Inspect local tokenizer model (first matching model under models/tokenizer)
    local_models = glob.glob('models/**/*tokenizer.model', recursive=True)
    assert local_models, 'No local tokenizer model found under models/'
    local_model_path = local_models[0]
    sp = spm.SentencePieceProcessor(model_file=local_model_path)
    local_piece_size = sp.get_piece_size()

    # Assertions
    if nemo_vocab_size is not None:
        assert nemo_vocab_size == local_piece_size, f"Tokenizer mismatch: .nemo vocab {nemo_vocab_size} != local tokenizer pieces {local_piece_size}"

    # Additionally ensure tokenizer can encode a Devanagari sample without all-UNK
    sample = "रोहन होड ज़ाल्लो!"
    ids = sp.encode(sample, out_type=int)
    assert isinstance(ids, list) and len(ids) > 0, 'Tokenization produced empty ids'
    assert any(i != sp.unk_id() for i in ids), f'All tokens are UNK for sample: {sample}'
