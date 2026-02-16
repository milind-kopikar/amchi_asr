#!/usr/bin/env python3
"""
Smoke test for AI4Bharat IndicConformer Marathi model.

This script is intended to be run inside WSL/Linux (or any Unix-like env).
It loads the pretrained AI4Bharat Marathi model, runs inference on the
`data/manifests/dev_small.json` manifest, and prints transcriptions and WER.

Purpose: verify model download, loading, and inference locally before doing
any cloud GPU fine-tuning.
"""

import os
import sys
import json
import traceback
import math

try:
    # Add uname shim for environments where os.uname is missing
    if not hasattr(os, "uname"):
        def _uname():
            import collections
            uname_result = collections.namedtuple('uname_result', ['sysname', 'nodename', 'release', 'version', 'machine'])
            return uname_result(sysname='Linux', nodename='localhost', release='0', version='0', machine='x86_64')
        os.uname = _uname

    import nemo.collections.asr as nemo_asr

    # Try to import jiwer.wer; if unavailable, provide a simple fallback WER.
    try:
        from jiwer import wer
    except Exception:
        def wer(ref, hyp):
            """A small word-level WER implementation fallback.

            Returns WER as a float in [0,1].
            """
            ref_words = ref.split()
            hyp_words = hyp.split()
            m = len(ref_words)
            n = len(hyp_words)
            # empty reference -> define WER as 0 if both empty else 1
            if m == 0:
                return 0.0 if n == 0 else 1.0
            # edit distance DP
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
            return dp[m][n] / float(m)
except Exception:
    print("Failed to import prerequisites. Ensure you're running inside WSL/Linux with the correct Python env and that `nemo-toolkit`, `jiwer` are installed.")
    traceback.print_exc()
    sys.exit(1)


def load_manifest(path):
    examples = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                examples.append(j)
            except Exception:
                # not JSON -> try TSV/other
                continue
    return examples


def main():
    # Prefer a small smoke test manifest if present (overrides default)
    manifest = os.path.join(os.path.dirname(__file__), '..', 'data', 'manifests', 'dev_small.json')
    manifest = os.path.normpath(manifest)
    smoke_manifest = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_smoke.jsonl')
    smoke_manifest = os.path.normpath(smoke_manifest)
    if os.path.exists(smoke_manifest):
        print(f"Using smoke manifest: {smoke_manifest}")
        manifest = smoke_manifest

    if not os.path.exists(manifest):
        print(f"Manifest not found at {manifest}")
        sys.exit(1)

    print("Loading AI4Bharat Marathi IndicConformer model (this may download files)...")
    # HF token should be in env var HF_TOKEN
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HF_TOKEN')
    # NeMo's package layout can vary between releases. Try a couple of import paths
    model = None
    def partial_restore_from_nemo(nemo_path):
        """Extract config and weights from a .nemo and load only matching-shape params."""
        import tarfile
        import tempfile
        import torch
        import yaml

        print(f"🔧 Attempting partial restore from: {nemo_path}")
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(nemo_path, 'r') as tar:
                # extract model_config.yaml and model_weights.ckpt
                members = {m.name: m for m in tar.getmembers()}
                if 'model_config.yaml' not in members:
                    raise RuntimeError('model_config.yaml missing in .nemo')
                if 'model_weights.ckpt' not in members:
                    raise RuntimeError('model_weights.ckpt missing in .nemo')
                tar.extract('model_config.yaml', path=td)
                tar.extract('model_weights.ckpt', path=td)

            config_path = os.path.join(td, 'model_config.yaml')
            ckpt_path = os.path.join(td, 'model_weights.ckpt')

            # load config
            with open(config_path, 'r') as f:
                conf = yaml.safe_load(f)

            # instantiate model from config (no weights)
            from nemo.collections.asr.models import ASRModel as _ASRModel
            try:
                model_instance = _ASRModel.from_config_dict(conf, trainer=None)
            except Exception as e:
                # fallback to attribute path
                model_instance = nemo_asr.models.ASRModel.from_config_dict(conf, trainer=None)

            # load checkpoint state dict
            ckpt = torch.load(ckpt_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)

            model_sd = model_instance.state_dict()
            filtered = {}
            matched = 0
            skipped = 0
            for k, v in state.items():
                if k in model_sd and list(v.shape) == list(model_sd[k].shape):
                    filtered[k] = v
                    matched += 1
                else:
                    skipped += 1
            print(f"🔁 Matched {matched} params, skipped {skipped} params")

            model_instance.load_state_dict(filtered, strict=False)
            return model_instance

    try:
        # Preferred: direct models module
        from nemo.collections.asr.models import ASRModel as _ASRModel
        # Prefer local patched .nemo if available (faster and avoids HF-model config problems)
        patched = os.path.join(os.path.dirname(__file__), '..', 'models', 'indicconformer_mr', 'indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo')
        patched = os.path.normpath(patched)
        if os.path.exists(patched):
            try:
                # Load from a local .nemo checkpoint directly (allow non-strict load to tolerate shape mismatches)
                model = _ASRModel.restore_from(patched, strict=False)
            except Exception as e:
                print('Restore_from failed with:', e)
                print('Falling back to partial state dict restore...')
                model = partial_restore_from_nemo(patched)
        else:
            model = _ASRModel.from_pretrained("ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large")
    except Exception:
        try:
            # Older/alternate layout: nemo.collections.asr.models attribute
            patched = os.path.join(os.path.dirname(__file__), '..', 'models', 'indicconformer_mr', 'indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo')
            patched = os.path.normpath(patched)
            if os.path.exists(patched):
                try:
                    model = nemo_asr.models.ASRModel.restore_from(patched, strict=False)
                except Exception as e:
                    print('Restore_from failed with:', e)
                    print('Falling back to partial state dict restore...')
                    model = partial_restore_from_nemo(patched)
            else:
                model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large")
        except Exception as e:
            print("Failed to load model:", e)
            traceback.print_exc()
            sys.exit(1)

    model.freeze()

    # Ensure language masks exist when we removed multilingual config:
    try:
        import torch
        if hasattr(model, 'ctc_decoder') and getattr(model.ctc_decoder, 'language_masks', None) is None:
            # Create a sensible language mask that selects ALL classes (avoid empty selection)
            try:
                # Prefer reading num classes from conv output or state_dict
                num_classes = None
                if hasattr(model.ctc_decoder, 'decoder_layers'):
                    # Conv1d output channels stores num classes
                    first_layer = list(model.ctc_decoder.decoder_layers)[0]
                    num_classes = getattr(first_layer, 'out_channels', None)
                if num_classes is None:
                    # Fallback: inspect state_dict
                    sdict = model.state_dict()
                    for k in sdict.keys():
                        if 'ctc_decoder.decoder_layers.0.weight' in k:
                            num_classes = sdict[k].shape[0]
                            break
                if num_classes is None:
                    num_classes = 256
                num_masks = 12
                # Create a plain list of class index tensors and rely on installed NeMo to handle tensor indices
                lm = [torch.arange(num_classes, dtype=torch.long) for _ in range(num_masks)]
                model.ctc_decoder.language_masks = lm
                print(f'⚙️ Injected default language_masks into ctc_decoder (select all {num_classes} classes)')
            except Exception as e:
                print('⚠️ Failed to construct full language_masks:', e)
    except Exception:
        print('Could not set language_masks; continuing')

    # Attach a debug wrapper to the CTC decoding path to capture shapes and lengths when decoding
    try:
        # Print some model-level debug info to help locate the CTC decoding implementation
        print('DEBUG: model attrs preview:', [a for a in dir(model) if not a.startswith('_')][:60])
        print('DEBUG: has ctc_decoder?', hasattr(model, 'ctc_decoder'), 'has ctc_decoding?', hasattr(model, 'ctc_decoding'))

        wrapped = False
        # 1) Try wrapping dec.ctc_decoding.ctc_decoder_predictions_tensor (older layout)
        dec = getattr(model, 'ctc_decoder', None)
        if dec is not None:
            ctc_decoding = getattr(dec, 'ctc_decoding', None)
            if ctc_decoding is not None and hasattr(ctc_decoding, 'ctc_decoder_predictions_tensor'):
                orig_fn = ctc_decoding.ctc_decoder_predictions_tensor
                def debug_ctc_decoder_predictions_tensor(*args, **kwargs):
                    try:
                        print('DEBUG: ctc_decoder_predictions_tensor called (via dec.ctc_decoding)')
                        if len(args) >= 2:
                            pred = args[0]
                            out_len = args[1]
                            try:
                                if isinstance(pred, (list, tuple)):
                                    print('DEBUG: prediction list len:', len(pred))
                                    if len(pred) > 0:
                                        p0 = pred[0]
                                        print('DEBUG: prediction[0] shape:', getattr(p0, 'shape', None), 'dtype:', getattr(p0, 'dtype', None))
                                else:
                                    print('DEBUG: prediction tensor shape:', getattr(pred, 'shape', None), 'dtype:', getattr(pred, 'dtype', None))
                            except Exception as d:
                                print('DEBUG: failed to inspect pred tensor:', d)
                            print('DEBUG: out_len:', out_len)

                        # Extra: print language_masks to detect dtype/shape issues before module-level handling
                        try:
                            ctc = getattr(model, 'ctc_decoder', None)
                            if ctc is not None:
                                lm = getattr(ctc, 'language_masks', None)
                                print('DEBUG: language_masks (post-inject) type:', type(lm), 'len:', len(lm) if hasattr(lm, '__len__') else None)
                                if isinstance(lm, (list, tuple)) and len(lm) > 0:
                                    first = lm[0]
                                    print('DEBUG: language_masks[0] type:', type(first), 'shape/dtype/numel:', getattr(first, 'shape', None), getattr(first, 'dtype', None), getattr(first, 'numel', lambda: None)())
                        except Exception as e:
                            print('DEBUG: failed to inspect language_masks (dec wrapper):', e)
                    except Exception as e:
                        print('DEBUG: error in ctc debug wrapper (dec.ctc_decoding):', e)
                    try:
                        return orig_fn(*args, **kwargs)
                    except Exception as e:
                        print('DEBUG: orig ctc_decoder_predictions_tensor raised:', type(e).__name__, e)
                        import traceback as _tb
                        _tb.print_exc()
                        # dump args info
                        try:
                            for i, a in enumerate(args[:3]):
                                try:
                                    print(f'DEBUG ARG[{i}] type:', type(a), 'repr:', repr(a)[:200])
                                except Exception:
                                    print(f'DEBUG ARG[{i}] repr failed')
                        except Exception:
                            pass
                        raise
                ctc_decoding.ctc_decoder_predictions_tensor = debug_ctc_decoder_predictions_tensor
                print('✅ Attached CTC debug wrapper: dec.ctc_decoding.ctc_decoder_predictions_tensor')
                wrapped = True

        # 2) Try wrapping model.ctc_decoding.ctc_decoder_predictions_tensor (alternative layout)
        if not wrapped and hasattr(model, 'ctc_decoding'):
            mcd = getattr(model, 'ctc_decoding')
            if hasattr(mcd, 'ctc_decoder_predictions_tensor'):
                orig_fn2 = mcd.ctc_decoder_predictions_tensor
                def debug_ctc_decoder_predictions_tensor2(*args, **kwargs):
                    try:
                        print('DEBUG: ctc_decoder_predictions_tensor called (via model.ctc_decoding)')
                        if len(args) >= 2:
                            pred = args[0]
                            out_len = args[1]
                            try:
                                if isinstance(pred, (list, tuple)):
                                    print('DEBUG: prediction list len:', len(pred))
                                    if len(pred) > 0:
                                        p0 = pred[0]
                                        print('DEBUG: prediction[0] shape:', getattr(p0, 'shape', None), 'dtype:', getattr(p0, 'dtype', None))
                                else:
                                    print('DEBUG: prediction tensor shape:', getattr(pred, 'shape', None), 'dtype:', getattr(pred, 'dtype', None))
                            except Exception as d:
                                print('DEBUG: failed to inspect pred tensor:', d)
                            print('DEBUG: out_len:', out_len)

                        # Extra: print language_masks details here as well
                        try:
                            lm = getattr(getattr(model, 'ctc_decoder', None), 'language_masks', None)
                            print('DEBUG: language_masks type:', type(lm), 'len:', len(lm) if hasattr(lm, '__len__') else None)
                            if isinstance(lm, (list, tuple)) and len(lm) > 0:
                                first = lm[0]
                                print('DEBUG: language_masks[0] type:', type(first), 'shape/dtype/numel:', getattr(first, 'shape', None), getattr(first, 'dtype', None), getattr(first, 'numel', lambda: None)())
                        except Exception as e:
                            print('DEBUG: failed to inspect language_masks (model wrapper):', e)
                    except Exception as e:
                        print('DEBUG: error in ctc debug wrapper (model.ctc_decoding):', e)
                    try:
                        return orig_fn2(*args, **kwargs)
                    except Exception as e:
                        print('DEBUG: orig model.ctc_decoder_predictions_tensor raised:', type(e).__name__, e)
                        import traceback as _tb
                        _tb.print_exc()
                        try:
                            for i, a in enumerate(args[:3]):
                                try:
                                    print(f'DEBUG ARG[{i}] type:', type(a), 'repr:', repr(a)[:200])
                                except Exception:
                                    print(f'DEBUG ARG[{i}] repr failed')
                        except Exception:
                            pass
                        raise
                mcd.ctc_decoder_predictions_tensor = debug_ctc_decoder_predictions_tensor2
                print('✅ Attached CTC debug wrapper: model.ctc_decoding.ctc_decoder_predictions_tensor')
                wrapped = True

        # 3) Fallback: try to import the module function and wrap it (last resort)
        if not wrapped:
            try:
                from nemo.collections.asr.parts.submodules import ctc_greedy_decoding as cgd
                if hasattr(cgd, 'ctc_decoder_predictions_tensor'):
                    orig_mod_fn = cgd.ctc_decoder_predictions_tensor
                    def debug_ctc_decoder_predictions_tensor_mod(*args, **kwargs):
                        try:
                            print('DEBUG: ctc_decoder_predictions_tensor called (module-level)')
                            if len(args) >= 2:
                                pred = args[0]
                                out_len = args[1]
                                try:
                                    if isinstance(pred, (list, tuple)):
                                        print('DEBUG: prediction list len:', len(pred))
                                        if len(pred) > 0:
                                            p0 = pred[0]
                                            print('DEBUG: prediction[0] shape:', getattr(p0, 'shape', None), 'dtype:', getattr(p0, 'dtype', None))
                                    else:
                                        print('DEBUG: prediction tensor shape:', getattr(pred, 'shape', None), 'dtype:', getattr(pred, 'dtype', None))
                                except Exception as d:
                                    print('DEBUG: failed to inspect pred tensor:', d)
                                print('DEBUG: out_len:', out_len)
                        except Exception as e:
                            print('DEBUG: error in ctc debug wrapper (module):', e)
                        try:
                            return orig_mod_fn(*args, **kwargs)
                        except Exception as e:
                            print('DEBUG: orig module ctc_decoder_predictions_tensor raised:', type(e).__name__, e)
                            import traceback as _tb
                            _tb.print_exc()
                            try:
                                for i, a in enumerate(args[:3]):
                                    try:
                                        print(f'DEBUG ARG[{i}] type:', type(a), 'repr:', repr(a)[:200])
                                    except Exception:
                                        print(f'DEBUG ARG[{i}] repr failed')
                            except Exception:
                                pass
                            raise
                    cgd.ctc_decoder_predictions_tensor = debug_ctc_decoder_predictions_tensor_mod
                    print('✅ Attached CTC debug wrapper: module ctc_greedy_decoding.ctc_decoder_predictions_tensor')
                    wrapped = True
            except Exception as e:
                print('⚠️ Module-level wrapping failed:', e)

        if not wrapped:
            print('⚠️ Could not attach any CTC wrapper; available nearby attrs:',
                  'dec.ctc_decoding:', hasattr(dec, 'ctc_decoding') if dec is not None else None,
                  'model.ctc_decoding:', hasattr(model, 'ctc_decoding'))

        # Print decoder-related config and parameter shapes to diagnose zero-class logits
        try:
            print('DEBUG: decoder attr:', getattr(model, 'decoder', None))
            print('DEBUG: decoder.num_classes:', getattr(getattr(model, 'decoder', None), 'num_classes', None))
            print('DEBUG: ctc_decoder:', getattr(model, 'ctc_decoder', None))
            print('DEBUG: ctc_decoder.num_classes:', getattr(getattr(model, 'ctc_decoder', None), 'num_classes', None))
            print('DEBUG: aux_ctc present?', hasattr(model, 'aux_ctc'))
            if hasattr(model, 'aux_ctc'):
                print('DEBUG: aux_ctc.decoder.num_classes:', getattr(getattr(model, 'aux_ctc', None).decoder, 'num_classes', None))
            # Print language_masks details
            try:
                lm = getattr(getattr(model, 'ctc_decoder', None), 'language_masks', None)
                print('DEBUG: ctc_decoder.language_masks type:', type(lm), 'value:', lm)
            except Exception as e:
                print('DEBUG: failed to inspect language_masks:', e)
            # Print shapes of relevant state dict entries
            sdict = model.state_dict()
            for k in sorted(sdict.keys()):
                kl = k.lower()
                if 'decoder' in kl or 'ctc' in kl or 'joint' in kl or 'prediction' in kl:
                    print('DEBUG PARAM', k, 'shape', list(sdict[k].shape))

            # --- Targeted instrumentation: detect empty/None vocabulary and fix ---
            try:
                # Inspect config-level vocabulary
                cfg_decoder_vocab = None
                try:
                    cfg_decoder_vocab = getattr(model.cfg, 'decoder', None)
                    cfg_vocab = None
                    if cfg_decoder_vocab is not None:
                        cfg_vocab = getattr(cfg_decoder_vocab, 'vocabulary', None)
                        print('DEBUG: cfg.decoder.vocabulary type:', type(cfg_vocab), 'len:', len(cfg_vocab) if cfg_vocab else None)
                except Exception as e:
                    print('DEBUG: could not read model.cfg.decoder.vocabulary:', e)

                # Inspect model-level decoder vocabulary attribute
                model_vocab = getattr(getattr(model, 'decoder', None), 'vocabulary', None)
                print('DEBUG: model.decoder.vocabulary type:', type(model_vocab), 'len:', len(model_vocab) if model_vocab else None)

                # If either is missing or empty, populate a dummy vocabulary of 256 tokens
                need_fix = False
                if not cfg_decoder_vocab or (hasattr(cfg_decoder_vocab, 'vocabulary') and (getattr(cfg_decoder_vocab, 'vocabulary') in (None, [], ''))):
                    need_fix = True
                if model_vocab in (None, [], ''):
                    need_fix = True

                if need_fix:
                    print('🔧 Detected missing/empty decoder vocabulary. Injecting dummy vocabulary of size 256')
                    dummy_size = 256
                    dummy_vocab = [f'token_{i}' for i in range(dummy_size)]
                    # set in cfg if present
                    if getattr(model, 'cfg', None) is not None:
                        try:
                            if not hasattr(model.cfg, 'decoder'):
                                model.cfg.decoder = {}
                            # For OmegaConf-like dicts, assign directly if supported
                            try:
                                model.cfg.decoder.vocabulary = dummy_vocab
                            except Exception:
                                # fallback: set attribute on runtime object
                                setattr(model.cfg.decoder, 'vocabulary', dummy_vocab)
                            print('🔧 Set model.cfg.decoder.vocabulary')
                        except Exception as e:
                            print('⚠️ Failed to set model.cfg.decoder.vocabulary:', e)

                    # set on model.decoder attribute (if exists)
                    if getattr(model, 'decoder', None) is not None:
                        try:
                            setattr(model.decoder, 'vocabulary', dummy_vocab)
                            print('🔧 Set model.decoder.vocabulary')
                        except Exception as e:
                            print('⚠️ Failed to set model.decoder.vocabulary:', e)

                    # Rely on the installed NeMo library for CTC conv shape and language mask handling.
                    # We intentionally do not perform runtime conv replacement or wrap language_masks here so that
                    # the smoke test validates the on-disk library changes instead of monkeypatching behavior.
                    print('ℹ️ Skipping runtime CTC conv replacement and language_mask wrapping; relying on installed library')
            except Exception as e:
                print('DEBUG: error during instrumentation/fix:', e)
        except Exception as e:
            print('DEBUG: failed to inspect decoder params:', e)
    except Exception as e:
        print('⚠️ Failed to attach CTC debug wrapper:', e)

    examples = load_manifest(manifest)
    if not examples:
        print("No examples found in manifest.")
        sys.exit(1)

    total_wer = 0.0
    count = 0

    print(f"Running inference on {len(examples)} examples...")
    for ex in examples:
        audio = ex.get('audio_filepath') or ex.get('audio')
        ref = ex.get('text', '')
        if not audio:
            continue
        # If the path is relative, make it relative to repo root
        if not os.path.isabs(audio):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            audio = os.path.normpath(os.path.join(repo_root, 'data', 'audio', os.path.basename(audio)))

        if not os.path.exists(audio):
            print(f"Audio file missing: {audio}")
            continue

        try:
            # Use CTC decoder for quick inference
            model.cur_decoder = 'ctc'
            # Use an integer language id to index into injected language_masks
            transcription = model.transcribe([audio], batch_size=1, logprobs=False, language_id=0)[0]
        except Exception as e:
            # Print full traceback for debugging the CTC decoding failure
            print(f"⚠️ Transcription failed for {audio}: {type(e).__name__}: {e}. Printing traceback:")
            import traceback as _tb
            _tb.print_exc()
            transcription = ""
            # proceed to print the (empty) transcription and compute WER if reference is present


        print("\n---")
        print(f"Audio: {audio}")
        print(f"Reference: {ref}")
        print(f"Transcription: {transcription}")

        if ref:
            try:
                e_wer = wer(ref, transcription)
                print(f"WER: {e_wer:.2%}")
                total_wer += e_wer
                count += 1
            except Exception:
                print("Failed to compute WER")

    if count:
        print(f"\nAverage WER on manifest: {total_wer/count:.2%}")


if __name__ == '__main__':
    main()
