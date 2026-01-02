#!/usr/bin/env python3
"""
Fine-tune AI4Bharat IndicConformer Marathi ASR model for Konkani
Using NVIDIA NeMo framework
"""

import sys
import os
import platform

# Only apply Windows patch if running on Windows
if platform.system() == 'Windows':
    sys.path.insert(0, os.path.dirname(__file__))
    import windows_patch

import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Research logger imports
import datetime, json, csv, time
from typing import Optional, List

# Simple WER helper for quick evaluations
def _compute_wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    m = len(r)
    n = len(h)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if r[i - 1] == h[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / m


def _compute_char_distance(ref: str, hyp: str) -> float:
    """Normalized character-level Levenshtein distance."""
    a = list(ref)
    b = list(hyp)
    m = len(a)
    n = len(b)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return float(dp[m][n]) / max(1, m)


class ResearchCSVLogger(pl.Callback):
    """Lightweight CSV logger that appends epoch metrics to a CSV file.

    Improvements:
    - Capture training loss at epoch end via on_train_epoch_end (fallbacks: 'train_loss', 'loss')
    - On validation end, try multiple metrics keys for val_loss and val_wer
    - Keep last seen training loss in memory so it is recorded even if trainer.callback_metrics doesn't expose it at validation end
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.start_time = time.time()
        self._last_train_loss = None

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        # Best-effort capture of training loss from a few possible sources:
        # 1) The `outputs` argument passed by Lightning (may contain per-batch outputs)
        # 2) trainer.callback_metrics keys ('train_loss', 'loss')
        try:
            # 1) Try extracting from outputs if available
            losses = []
            def _extract(o):
                if o is None:
                    return
                if isinstance(o, dict):
                    for k in ('train_loss', 'loss', 'loss_value'):
                        if k in o:
                            v = o[k]
                            try:
                                # handle tensors
                                if hasattr(v, 'detach'):
                                    losses.append(float(v.detach().cpu()))
                                else:
                                    losses.append(float(v))
                            except Exception:
                                pass
                    # there may be nested lists/dicts
                    for v in o.values():
                        _extract(v)
                elif isinstance(o, (list, tuple)):
                    for e in o:
                        _extract(e)
            _extract(outputs)

            if losses:
                self._last_train_loss = float(sum(losses) / len(losses))
            else:
                # 2) fallback to trainer callback metrics
                metrics = trainer.callback_metrics
                train_loss = metrics.get('train_loss', None) or metrics.get('loss', None) or metrics.get('training_loss', None)
                self._last_train_loss = float(train_loss) if train_loss is not None else None
        except Exception:
            # Best-effort; do not fail the epoch
            pass

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip writing metrics during the trainer's sanity check phase (avoids spurious empty rows)
        if getattr(trainer, 'sanity_checking', False):
            return

        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        # Train loss: prefer last captured train epoch loss, then fall back to common metric keys
        train_loss = self._last_train_loss
        if train_loss is None:
            # Sometimes PTL exposes aggregated epoch metrics under different keys; try several
            train_loss = metrics.get('train_loss', None) or metrics.get('loss', None) or metrics.get('training_loss', None) or metrics.get('train/loss', None)

        # Validation loss: try a few common keys
        val_loss = metrics.get('val_loss', None) or metrics.get('validation_loss', None) or metrics.get('val/loss', None) or metrics.get('val_epoch_loss', None)
        val_wer = metrics.get('val_wer', None) or metrics.get('val_wer_ctc', None) or metrics.get('validation_wer', None) or metrics.get('val/wer', None)
        val_char = metrics.get('val_char_dist', None) or metrics.get('val_char', None)

        lr = None
        try:
            opt = trainer.optimizers[0]
            lr = opt.param_groups[0].get('lr', None)
        except Exception:
            pass
        time_elapsed = time.time() - self.start_time
        try:
            with open(self.filepath, 'a', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow([epoch,
                                 float(train_loss) if train_loss is not None else '',
                                 float(val_loss) if val_loss is not None else '',
                                 float(val_wer) if val_wer is not None else '',
                                 float(val_char) if val_char is not None else '',
                                 lr if lr is not None else '',
                                 round(time_elapsed, 2)])
        except Exception as e:
            logger.warning(f"Failed to append CSV epoch metrics: {e}")


class SampleLoggerCallback(pl.Callback):
    """Saves a few validation sample transcriptions per epoch into JSON files."""
    def __init__(self, manifest_path: str, outdir: str, max_samples: int = 8):
        self.manifest_path = manifest_path
        self.outdir = outdir
        self.max_samples = max_samples
        self.samples = []
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as fh:
                for i, line in enumerate(fh):
                    if i >= self.max_samples:
                        break
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self.samples.append({'audio': obj.get('audio_filepath'), 'reference': obj.get('text', '')})
        except Exception as e:
            logger.warning(f"Failed to read manifest for SampleLogger: {e}")


class ForceLRCallback(pl.Callback):
    """Force optimizer learning rate to a fixed value at training start.

    This ensures the actual optimizer param_groups['lr'] matches the config value
    even if model.configure_optimizers or other code attempts to set a different base lr.
    """
    def __init__(self, lr: float):
        self.lr = float(lr)
    def on_train_start(self, trainer, pl_module):
        try:
            for opt in getattr(trainer, 'optimizers', []):
                for g in opt.param_groups:
                    g['lr'] = self.lr
            logger.info(f"Forced optimizer lr to {self.lr}")
        except Exception as e:
            logger.warning(f"Failed to force optimizer lr: {e}")

    # Silence validation hooks to avoid accidental calls (no-op)
    def on_validation_epoch_end(self, trainer, pl_module):
        return

    def on_train_end(self, trainer, pl_module):
        return

    def _is_devanagari(self, text: str) -> bool:
        if not text:
            return False
        for ch in text:
            o = ord(ch)
            # Devanagari block U+0900..U+097F
            if 0x0900 <= o <= 0x097F:
                return True
        return False


class SampleLoggerDebug(pl.Callback):
    """Debug replacement for SampleLoggerCallback — logs and writes sample predictions with extra debug prints."""
    def __init__(self, manifest_path: str, outdir: str, max_samples: int = 8):
        self.manifest_path = manifest_path
        self.outdir = outdir
        self.max_samples = max_samples
        self.samples = []
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as fh:
                for i, line in enumerate(fh):
                    if i >= self.max_samples:
                        break
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self.samples.append({'audio': obj.get('audio_filepath'), 'reference': obj.get('text', '')})
        except Exception as e:
            logger.warning(f"Failed to read manifest for SampleLoggerDebug: {e}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not hasattr(self, 'samples'):
            return
        epoch = trainer.current_epoch
        global_step = getattr(trainer, 'global_step', None)
        audio_paths = [s['audio'] for s in getattr(self, 'samples', []) if s.get('audio')]
        preds = []
        start_t = time.time()
        try:
            print("DEBUG: SampleLogger started", flush=True)
            logger.info(f"DEBUG: SampleLogger started - samples_count={len(self.samples)}, audio_count={len(audio_paths)}")
        except Exception:
            pass

        def _normalize_pred(p):
            import ast
            if isinstance(p, str):
                try:
                    parsed = ast.literal_eval(p)
                    return _normalize_pred(parsed)
                except Exception:
                    return p
            if isinstance(p, (list, tuple)) and len(p) > 0:
                first = p[0]
                if isinstance(first, (list, tuple)) and len(first) > 0:
                    return _normalize_pred(first[0])
                return _normalize_pred(first)
            return str(p)

        def _map_language_id(mod, lid_str):
            mapped = None
            try:
                if hasattr(mod, 'joint') and hasattr(mod.joint, 'language_keys'):
                    mapped = list(mod.joint.language_keys).index(lid_str)
            except Exception:
                mapped = None
            if mapped is None:
                try:
                    if hasattr(mod, 'cfg') and 'language_keys' in getattr(mod.cfg, 'joint', {}):
                        mapped = list(mod.cfg.joint.language_keys).index(lid_str)
                except Exception:
                    mapped = None
            return mapped

        try:
            if hasattr(pl_module, 'transcribe'):
                prev_decoder = getattr(pl_module, 'cur_decoder', None)
                try:
                    pl_module.cur_decoder = 'ctc'
                except Exception:
                    pass
                try:
                    for audio in audio_paths:
                        got = None
                        candidate_kwargs = [ {'language_id': 'kok', 'batch_size': 1, 'logprobs': False}, {'batch_size': 1, 'logprobs': False}, {'logprobs': False}, {} ]
                        for kw in candidate_kwargs:
                            try:
                                kwargs = dict(kw)
                                if 'language_id' in kwargs and isinstance(kwargs['language_id'], str):
                                    mapped = _map_language_id(pl_module, kwargs['language_id'])
                                    if mapped is not None:
                                        kwargs['language_id'] = mapped
                                try:
                                    out = pl_module.transcribe([audio], **kwargs)
                                except TypeError:
                                    try:
                                        out = pl_module.transcribe(paths2audio_files=[audio], **kwargs)
                                    except TypeError:
                                        out = pl_module.transcribe(audio, **kwargs)
                                got = _normalize_pred(out)
                                if isinstance(got, str) and got != '':
                                    break
                            except Exception:
                                got = None
                                continue
                        preds.append(got if got is not None else '')
                finally:
                    try:
                        if prev_decoder is None:
                            if hasattr(pl_module, 'cur_decoder'):
                                delattr(pl_module, 'cur_decoder')
                        else:
                            pl_module.cur_decoder = prev_decoder
                    except Exception:
                        pass
            else:
                preds = [''] * len(audio_paths)
        except Exception as e:
            logger.warning(f"SampleLoggerDebug failed to run inference: {e}")

        end_t = time.time()
        batch_time = end_t - start_t

        results = []
        wer_list = []
        for i, s in enumerate(self.samples):
            audio = s.get('audio')
            ref = s.get('reference', '')
            pred = preds[i] if i < len(preds) else ''
            try:
                if i == 0:
                    print(f"DEBUG PREDICTION: {pred}", flush=True)
                    logger.info(f"DEBUG PREDICTION: {pred}")
            except Exception:
                pass
            wer = _compute_wer(ref, pred) if ref and pred is not None else None
            if wer is not None:
                wer_list.append(wer)
            # char-level distance
            char_d = None
            try:
                char_d = _compute_char_distance(ref, pred) if ref is not None and pred is not None else None
            except Exception:
                char_d = None
            if char_d is not None:
                char_list.append(char_d)
            pred_latency_s = (batch_time / max(1, len(audio_paths))) if audio_paths else None
            deva_ok = any(0x0900 <= ord(ch) <= 0x097F for ch in pred) if pred else False
            results.append({'index': i, 'audio': audio, 'ref': ref, 'pred': pred, 'deva_ok': bool(deva_ok), 'pred_latency_s': pred_latency_s, 'wer': wer, 'char_dist': char_d})

        overall_wer = sum(wer_list) / max(1, len(wer_list)) if wer_list else None
        overall_char = sum(char_list) / max(1, len(char_list)) if char_list else None
        try:
            if overall_wer is not None:
                try:
                    if hasattr(pl_module, 'log'):
                        pl_module.log('val_wer', float(overall_wer), on_epoch=True, on_step=False)
                except Exception:
                    pass
                try:
                    trainer.callback_metrics['val_wer'] = float(overall_wer)
                except Exception:
                    pass
            if overall_char is not None:
                try:
                    if hasattr(pl_module, 'log'):
                        pl_module.log('val_char_dist', float(overall_char), on_epoch=True, on_step=False)
                except Exception:
                    pass
                try:
                    trainer.callback_metrics['val_char_dist'] = float(overall_char)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to log aggregate val_wer/val_char: {e}")

        out = {'summary': {'epoch': int(epoch), 'global_step': int(global_step) if global_step is not None else None, 'model_name': pl_module.__class__.__name__, 'param_count': int(pl_module.num_parameters()) if hasattr(pl_module, 'num_parameters') else None, 'overall_wer': overall_wer, 'date': datetime.datetime.now().isoformat()}, 'samples': results}
        outpath = os.path.join(self.outdir, f'samples_epoch_{epoch:02d}.json')
        try:
            print(f"DEBUG: Attempting to write file to {outpath}", flush=True)
            logger.info(f"DEBUG: Attempting to write file to {outpath}")
        except Exception:
            pass
        try:
            os.makedirs(self.outdir, exist_ok=True)
            with open(outpath, 'w', encoding='utf-8') as fh:
                json.dump(out, fh, indent=2, ensure_ascii=False)
            print(f"DEBUG: Wrote sample results to {outpath}", flush=True)
            logger.info(f"DEBUG: Wrote sample results to {outpath}")
        except Exception as e:
            logger.warning(f"Failed to write sample results to {outpath}: {e}")


class SampleLoggerWriter(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Guard against accidental invocation on unrelated callback instances
        if not hasattr(self, 'samples'):
            return

        epoch = trainer.current_epoch
        global_step = getattr(trainer, 'global_step', None)

        audio_paths = [s['audio'] for s in getattr(self, 'samples', []) if s.get('audio')]
        preds = []
        start_t = time.time()
        preds = []
        try:
            # Debug: indicate the logger started and how many samples
            try:
                print("DEBUG: SampleLogger started", flush=True)
                print(f"DEBUG: samples_count={len(self.samples)}, audio_count={len(audio_paths)}", flush=True)
                try:
                    logger.info(f"DEBUG: SampleLogger started - samples_count={len(self.samples)}, audio_count={len(audio_paths)}")
                except Exception:
                    pass
            except Exception:
                pass
            # Use a robust per-sample transcribe strategy similar to the smoke test.
            # Try mapping a string language id to an index if needed, and attempt
            # multiple transcribe signatures and kwargs to ensure we get a usable string.
            def _normalize_pred(p):
                import ast
                # If it's a string representation of a Python object, try to parse
                if isinstance(p, str):
                    try:
                        parsed = ast.literal_eval(p)
                        return _normalize_pred(parsed)
                    except Exception:
                        return p
                if isinstance(p, (list, tuple)) and len(p) > 0:
                    first = p[0]
                    if isinstance(first, (list, tuple)) and len(first) > 0:
                        return _normalize_pred(first[0])
                    return _normalize_pred(first)
                return str(p)

            # Helper to map a string language_id -> index when model expects numeric ids
            def _map_language_id(mod, lid_str):
                mapped = None
                try:
                    if hasattr(mod, 'joint') and hasattr(mod.joint, 'language_keys'):
                        mapped = list(mod.joint.language_keys).index(lid_str)
                except Exception:
                    mapped = None
                if mapped is None:
                    try:
                        if hasattr(mod, 'cfg') and 'language_keys' in getattr(mod.cfg, 'joint', {}):
                            mapped = list(mod.cfg.joint.language_keys).index(lid_str)
                    except Exception:
                        mapped = None
                return mapped

            if hasattr(pl_module, 'transcribe'):
                # Force CTC decoder during these short validation calls to avoid using an untrained RNNT decoder
                prev_decoder = getattr(pl_module, 'cur_decoder', None)
                try:
                    try:
                        pl_module.cur_decoder = 'ctc'
                    except Exception:
                        pass

                    for audio in audio_paths:
                        got = None
                        # Try options: prefer passing batch/list but fall back to singular forms
                        candidate_kwargs = [ {'language_id': 'kok', 'batch_size': 1, 'logprobs': False}, {'batch_size': 1, 'logprobs': False}, {'logprobs': False}, {} ]
                        for kw in candidate_kwargs:
                            try:
                                kwargs = dict(kw)
                                if 'language_id' in kwargs and isinstance(kwargs['language_id'], str):
                                    mapped = _map_language_id(pl_module, kwargs['language_id'])
                                    if mapped is not None:
                                        kwargs['language_id'] = mapped
                                # Try common call forms
                                try:
                                    out = pl_module.transcribe([audio], **kwargs)
                                except TypeError:
                                    try:
                                        out = pl_module.transcribe(paths2audio_files=[audio], **kwargs)
                                    except TypeError:
                                        out = pl_module.transcribe(audio, **kwargs)
                                got = _normalize_pred(out)
                                # Acceptable result (non-empty)
                                if isinstance(got, str) and got != '':
                                    break
                            except Exception:
                                # try next kw / call form
                                got = None
                                continue
                        preds.append(got if got is not None else '')
                finally:
                    # restore previous decoder if it existed
                    try:
                        if prev_decoder is None:
                            if hasattr(pl_module, 'cur_decoder'):
                                delattr(pl_module, 'cur_decoder')
                        else:
                            pl_module.cur_decoder = prev_decoder
                    except Exception:
                        pass
            else:
                preds = [''] * len(audio_paths)
        except Exception as e:
            logger.warning(f"SampleLoggerCallback failed to run inference: {e}")
        end_t = time.time()
        batch_time = end_t - start_t

        results = []
        wer_list = []
        char_list = []
        for i, s in enumerate(self.samples):
            audio = s.get('audio')
            ref = s.get('reference', '')
            pred = preds[i] if i < len(preds) else ''
            # Debug: print the first prediction to stderr/stdout so we see it even if file write fails
            try:
                if i == 0:
                    print(f"DEBUG PREDICTION: {pred}", flush=True)
                    try:
                        logger.info(f"DEBUG PREDICTION: {pred}")
                    except Exception:
                        pass
            except Exception:
                pass
            wer = _compute_wer(ref, pred) if ref and pred is not None else None
            if wer is not None:
                wer_list.append(wer)
            # char-level distance
            char_d = None
            try:
                char_d = _compute_char_distance(ref, pred) if ref is not None and pred is not None else None
            except Exception:
                char_d = None
            if char_d is not None:
                char_list.append(char_d)
            pred_latency_s = (batch_time / max(1, len(audio_paths))) if audio_paths else None
            deva_ok = self._is_devanagari(pred)
            results.append({
                'index': i,
                'audio': audio,
                'ref': ref,
                'pred': pred,
                'deva_ok': bool(deva_ok),
                'pred_latency_s': pred_latency_s,
                'wer': wer,
                'char_dist': char_d
            })

        # Build summary block
        # model_name: prefer config name if present, fall back to class name
        try:
            cfg_name = None
            if hasattr(pl_module, '_cfg') and isinstance(pl_module._cfg, dict) and 'name' in pl_module._cfg:
                cfg_name = pl_module._cfg['name']
            elif hasattr(pl_module, '_cfg') and hasattr(pl_module._cfg, 'name'):
                cfg_name = getattr(pl_module._cfg, 'name')
        except Exception:
            cfg_name = None

        model_name = cfg_name if cfg_name else pl_module.__class__.__name__

        # param_count
        try:
            if hasattr(pl_module, 'num_parameters') and callable(getattr(pl_module, 'num_parameters')):
                param_count = int(pl_module.num_parameters())
            elif hasattr(pl_module, 'num_parameters'):
                param_count = int(getattr(pl_module, 'num_parameters'))
            else:
                param_count = int(sum([p.numel() for p in pl_module.parameters()]))
        except Exception:
            param_count = None

        overall_wer = sum(wer_list) / max(1, len(wer_list)) if wer_list else None
        overall_char = sum(char_list) / max(1, len(char_list)) if char_list else None

        # Log aggregate validation WER and char distance so ModelCheckpoint can monitor them if requested
        try:
            if overall_wer is not None:
                try:
                    if hasattr(pl_module, 'log'):
                        pl_module.log('val_wer', float(overall_wer), on_epoch=True, on_step=False)
                except Exception:
                    pass
                # As a more direct and immediate fallback, inject val_wer into trainer.callback_metrics
                try:
                    trainer.callback_metrics['val_wer'] = float(overall_wer)
                except Exception:
                    pass
            if overall_char is not None:
                try:
                    if hasattr(pl_module, 'log'):
                        pl_module.log('val_char_dist', float(overall_char), on_epoch=True, on_step=False)
                except Exception:
                    pass
                try:
                    trainer.callback_metrics['val_char_dist'] = float(overall_char)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Failed to log aggregate val_wer/val_char: {e}")

        out = {
            'summary': {
                'epoch': int(epoch),
                'global_step': int(global_step) if global_step is not None else None,
                'model_name': model_name,
                'param_count': param_count,
                'overall_wer': overall_wer,
                'date': datetime.datetime.now().isoformat()
            },
            'samples': results
        }

        # Duplicate SampleLogger write block removed from ForceLRCallback; this method is a no-op here.
        return

# NeMo imports
import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils import exp_manager
from nemo.utils.exceptions import NeMoBaseException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> DictConfig:
    """
    Load and validate configuration

    Args:
        config_path: Path to YAML configuration file

    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = OmegaConf.load(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    return config

def setup_model(config: DictConfig):
    """
    Setup the ASR model for fine-tuning

    Args:
        config: Configuration object

    Returns:
        NeMo ASR model
    """
    logger.info("Setting up ASR model...")

    # Optionally skip restoring from .nemo (useful for fast micro-overfit/testing)
    if os.environ.get('MICRO_SKIP_MODEL_RESTORE', '0') == '1':
        # Try to instantiate model from a local model_config.yaml if present
        local_mc = 'model_config.yaml'
        if os.path.exists(local_mc):
            import yaml
            logger.info('MICRO_SKIP_MODEL_RESTORE=1 -> instantiating model from local model_config.yaml')
            with open(local_mc, 'r', encoding='utf-8') as fh:
                conf = yaml.safe_load(fh)
            try:
                model = nemo_asr.models.ASRModel.from_config_dict(conf, trainer=None)
            except Exception:
                from nemo.collections.asr.models import ASRModel as _ASRModel
                model = _ASRModel.from_config_dict(conf, trainer=None)
        else:
            raise RuntimeError('MICRO_SKIP_MODEL_RESTORE=1 but local model_config.yaml not found')
        
    else:
        # Load base model
        model_path = config.model.nemo_model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading base model: {model_path}")
        try:
            model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path, strict=False)
        except Exception as e:
        # Fallback: attempt partial restore loading only matching-shape params
        import tarfile, tempfile, torch, yaml
        print(f"🔧 Partial restore from {model_path} due to: {e}")
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(model_path, 'r') as tar:
                members = {m.name: m for m in tar.getmembers()}
                if 'model_config.yaml' not in members or 'model_weights.ckpt' not in members:
                    raise RuntimeError('model_config.yaml or model_weights.ckpt missing in .nemo')
                tar.extract('model_config.yaml', path=td)
                tar.extract('model_weights.ckpt', path=td)

            config_path = os.path.join(td, 'model_config.yaml')
            ckpt_path = os.path.join(td, 'model_weights.ckpt')
            with open(config_path, 'r') as f:
                conf = yaml.safe_load(f)

            try:
                model = nemo_asr.models.ASRModel.from_config_dict(conf, trainer=None)
            except Exception:
                from nemo.collections.asr.models import ASRModel as _ASRModel
                model = _ASRModel.from_config_dict(conf, trainer=None)

            ckpt = torch.load(ckpt_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            model_sd = model.state_dict()
            filtered = {}
            matched = skipped = 0
            for k, v in state.items():
                if k in model_sd and list(v.shape) == list(model_sd[k].shape):
                    filtered[k] = v
                    matched += 1
                else:
                    skipped += 1
            print(f"🔁 Matched {matched} params, skipped {skipped} params")
            model.load_state_dict(filtered, strict=False)

    # Freeze encoder layers (optional - for faster training with less data)
    if hasattr(config, 'freeze_encoder') and config.freeze_encoder:
        logger.info("Freezing encoder layers...")
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Runtime patching and safety measures for pilot runs:
    # 1) Optionally apply the vendored conv_asr patch to the installed NeMo module so
    #    ConvASRDecoder can accept string language ids like 'kok'. This avoids
    #    altering the base package on disk and is safe for pilots. Set
    #    `APPLY_CONV_PATCH=1` in the environment to enable.
    try:
        if os.environ.get('APPLY_CONV_PATCH') == '1':
            try:
                # Ensure repo root is importable so `patches` can be imported from anywhere
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                # Import the vendored patch module and monkey-patch the installed module
                import patches.conv_asr_fixed as patched_conv
                import nemo.collections.asr.modules.conv_asr as orig_conv
                # Replace the classes in-place
                orig_conv.ConvASRDecoder = patched_conv.ConvASRDecoder
                orig_conv.ConvASREncoder = patched_conv.ConvASREncoder
                orig_conv.ConvASRDecoderClassification = patched_conv.ConvASRDecoderClassification
                logger.info('Applied conv_asr runtime patch (patched ConvASRDecoder/Encoder)')
            except Exception as e:
                logger.warning(f'Failed to apply conv_asr patch at runtime: {e}')
    except Exception:
        pass

    # 2) Temporary stub replacement (ONLY if explicitly requested via USE_CTC_STUB=1)
    #    This is unsafe for research results; do not enable unless debugging.
    try:
        import torch
        if os.environ.get('USE_CTC_STUB') == '1' and hasattr(model, 'ctc_decoder'):
            try:
                # Validate a dry-run forward to see if ctc_decoder accepts our language id form
                dummy_in = torch.randn(1, model.preprocessor.features, 16)
                _ = model.ctc_decoder(dummy_in, language_ids=[getattr(config, 'custom_config', {}).get('language', 'kok')])
            except Exception as e:
                logger.warning(f"CTC decoder raised during dry-run: {e}. Replacing with a safe stub for pilot runs (USE_CTC_STUB=1).")

                class _DummyCTCDecoder(torch.nn.Module):
                    def __init__(self, num_classes_with_blank):
                        super().__init__()
                        self._num_classes_with_blank = num_classes_with_blank
                    def forward(self, encoder_output, language_ids=None):
                        # encoder_output: [B, C, T] expected by ConvASRDecoder; convert to [B, T, C]
                        if encoder_output.dim() == 3:
                            b, c, t = encoder_output.shape
                            out = torch.zeros((b, t, self._num_classes_with_blank), device=encoder_output.device, dtype=encoder_output.dtype)
                        else:
                            out = torch.zeros((1, 1, self._num_classes_with_blank), device=next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else 'cpu')
                        return out

                try:
                    num_classes = getattr(model.ctc_decoder, 'num_classes_with_blank', None)
                    if num_classes is None:
                        sdict = model.state_dict()
                        num_classes = 256
                        for k in sdict.keys():
                            if 'ctc_decoder.decoder_layers.0.weight' in k:
                                num_classes = sdict[k].shape[0]
                                break
                    model.ctc_decoder = _DummyCTCDecoder(num_classes)
                    logger.info('Replaced model.ctc_decoder with safe stub for pilot run (USE_CTC_STUB=1)')
                except Exception as e2:
                    logger.warning(f'Failed to replace ctc_decoder: {e2}')
    except Exception:
        pass

    # Update model configuration for fine-tuning
    if hasattr(config.model, 'decoder'):
        # Update decoder vocabulary if needed
        try:
            # If the decoder vocabulary is missing in the config or model, try to populate it from
            # the local SentencePiece tokenizer so we don't fall back to a dummy 256-token vocab.
            import sentencepiece as spm
            tk_dir = getattr(config.model, 'tokenizer', {}).get('dir', None) if hasattr(config.model, 'tokenizer') else None
            sp_model_path = None
            if tk_dir and os.path.isdir(tk_dir):
                for f in os.listdir(tk_dir):
                    if f.endswith('.model'):
                        sp_model_path = os.path.join(tk_dir, f)
                        break
            if sp_model_path is not None:
                sp = spm.SentencePieceProcessor(model_file=sp_model_path)
                pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

                cfg_decoder = getattr(config.model, 'decoder', None)
                try:
                    cfg_vocab = cfg_decoder.vocabulary if cfg_decoder is not None and hasattr(cfg_decoder, 'vocabulary') else None
                except Exception:
                    cfg_vocab = None

                model_vocab = None
                try:
                    model_vocab = getattr(getattr(model, 'decoder', None), 'vocabulary', None)
                except Exception:
                    model_vocab = None

                if not cfg_vocab or cfg_vocab in (None, [], '') or model_vocab in (None, [], ''):
                    logger.info('Populating decoder vocabulary from SentencePiece tokenizer (size=%d) to avoid dummy 256 fill-in', len(pieces))
                    try:
                        # set in config where possible
                        if cfg_decoder is not None:
                            try:
                                cfg_decoder.vocabulary = pieces
                            except Exception:
                                try:
                                    setattr(config.model.decoder, 'vocabulary', pieces)
                                except Exception:
                                    pass
                        # set on model runtime attribute
                        if getattr(model, 'decoder', None) is not None:
                            try:
                                setattr(model.decoder, 'vocabulary', pieces)
                            except Exception:
                                pass
                    except Exception as e:
                        logger.warning(f'Failed to populate decoder vocabulary from tokenizer: {e}')
        except Exception:
            # If sentencepiece is not available or fails, skip this best-effort step
            pass

    # If the user explicitly requested CTC via config (decoder_type or loss_name),
    # force the model to use CTC loss to avoid RNNT/Numba GPU JIT compilation.
    try:
        loss_choice = getattr(config, 'loss', {}).get('loss_name', None) if hasattr(config, 'loss') else None
        decoder_choice = getattr(config.model, 'decoder_type', None) if hasattr(config, 'model') else None
        if (loss_choice == 'ctc') or (decoder_choice == 'ctc'):
            if hasattr(model, 'ctc_loss'):
                model.loss = model.ctc_loss
                logger.info('Switched model.loss to ctc_loss (CTC) to avoid RNNT/Numba GPU JIT issues')
                # Update internal cfg if present
                try:
                    if hasattr(model, '_cfg') and 'loss' in model._cfg:
                        model._cfg.loss.loss_name = 'ctc'
                except Exception:
                    pass
            else:
                logger.warning('CTC loss not present on model; cannot force CTC. Proceeding with model default loss.')
    except Exception as e:
        logger.warning(f'Failed to enforce CTC loss via config: {e}')

    # If CTC was requested, monkey-patch a CTC-only training_step to avoid RNNT/jit codepath
    try:
        loss_choice = getattr(config, 'loss', {}).get('loss_name', None) if hasattr(config, 'loss') else None
        decoder_choice = getattr(config.model, 'decoder_type', None) if hasattr(config, 'model') else None
        if (loss_choice == 'ctc') or (decoder_choice == 'ctc'):
            # enforce decoder selection
            try:
                model.cur_decoder = 'ctc'
            except Exception:
                pass

            # Replace training_step with a CTC-only variant to skip RNNT joint computation
            def _ctc_training_step(self, batch, batch_nb):
                import torch
                # Unpack batch
                try:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 6:
                        signal, signal_len, transcript, transcript_len, sample_ids, language_ids = batch
                    elif isinstance(batch, dict):
                        signal = batch['input_signal']
                        signal_len = batch['input_signal_length']
                        transcript = batch['labels']
                        transcript_len = batch['labels_length']
                        language_ids = batch.get('language_ids', None)
                    else:
                        # fallback: try standard unpack
                        signal, signal_len, transcript, transcript_len = batch
                        language_ids = None
                except Exception:
                    # raise to let PL handle abnormal batch formats
                    raise

                # Encoder forward
                encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)

                # CTC logits and loss
                if "multisoftmax" not in getattr(self, 'cfg', {}).get('decoder', {}):
                    lang_ids = None
                else:
                    lang_ids = language_ids

                log_probs = self.ctc_decoder(encoder_output=encoded, language_ids=lang_ids)
                ctc_loss = self.ctc_loss(log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len)

                # Add auxiliary losses if any
                loss_value = self.add_auxiliary_losses(ctc_loss) if hasattr(self, 'add_auxiliary_losses') else ctc_loss

                # Log train loss in a way PyTorch Lightning aggregates per epoch
                try:
                    # on_epoch=True to ensure Train epoch aggregated metric is available in callback_metrics at validation end
                    self.log('train_loss', loss_value, on_epoch=True, on_step=False, prog_bar=False)
                except Exception:
                    pass

                # Optional logging (legacy)
                tensorboard_logs = {'loss': loss_value}
                return {'loss': loss_value, 'log': tensorboard_logs}

            # Validation step to compute val_loss and log it (helps CSV capture val_loss reliably)
            def _ctc_validation_step(self, batch, batch_nb):
                import torch
                # Unpack batch
                try:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 6:
                        signal, signal_len, transcript, transcript_len, sample_ids, language_ids = batch
                    elif isinstance(batch, dict):
                        signal = batch['input_signal']
                        signal_len = batch['input_signal_length']
                        transcript = batch['labels']
                        transcript_len = batch['labels_length']
                        language_ids = batch.get('language_ids', None)
                    else:
                        # fallback: try standard unpack
                        signal, signal_len, transcript, transcript_len = batch
                        language_ids = None
                except Exception:
                    raise

                encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
                if "multisoftmax" not in getattr(self, 'cfg', {}).get('decoder', {}):
                    lang_ids = None
                else:
                    lang_ids = language_ids
                log_probs = self.ctc_decoder(encoder_output=encoded, language_ids=lang_ids)
                val_loss = self.ctc_loss(log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len)
                try:
                    self.log('val_loss', val_loss, on_epoch=True, on_step=False, prog_bar=False)
                except Exception:
                    pass

                # Best-effort batch WER approximation via greedy CTC argmax decoding (token-based)
                try:
                    # Greedy decode: argmax over class dim
                    pred_ids = torch.argmax(log_probs, dim=-1)  # [B, T]
                    def _collapse_ids(row):
                        out = []
                        prev = None
                        for t in row.tolist():
                            if t == prev:
                                continue
                            prev = t
                            # treat 0 as blank id (best-effort)
                            if t == 0:
                                continue
                            out.append(str(int(t)))
                        return ' '.join(out)
                    preds_tok = [_collapse_ids(r) for r in pred_ids]
                    refs_tok = []
                    for r in transcript:
                        if hasattr(r, 'tolist'):
                            refs_tok.append(' '.join([str(int(x)) for x in r.tolist()]))
                        else:
                            refs_tok.append(str(r))
                    # compute token-level WER as proxy (use _compute_wer on space-separated token ids)
                    wer_vals = []
                    for ref, pred in zip(refs_tok, preds_tok):
                        if ref is None or pred is None:
                            continue
                        try:
                            wer_vals.append(_compute_wer(ref, pred))
                        except Exception:
                            pass
                    if wer_vals:
                        batch_wer = float(sum(wer_vals) / len(wer_vals))
                        try:
                            self.log('val_wer', batch_wer, on_epoch=True, on_step=False, prog_bar=False)
                        except Exception:
                            pass
                except Exception:
                    pass

                return {'val_loss': val_loss}

            import types
            model.training_step = types.MethodType(_ctc_training_step, model)
            model.validation_step = types.MethodType(_ctc_validation_step, model)
            logger.info('Patched model.training_step and model.validation_step to CTC-only implementations for pilot runs')
    except Exception as e:
        logger.warning(f'Failed to monkey-patch training step for CTC: {e}')

    logger.info("Model setup completed")
    return model

def setup_data_module(config: DictConfig, model=None):
    """
    Setup data module for training

    Args:
        config: Configuration object

    Returns:
        Data module
    """
    logger.info("Setting up data module...")

    # Create data module using the AudioToBPEDataset in the audio_to_text submodule if available
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
    import os, json
    tokenizer_dir = config.model.tokenizer.dir

    # find sentencepiece model
    sp_model = None
    if os.path.isdir(tokenizer_dir):
        for f in os.listdir(tokenizer_dir):
            if f.endswith('_tokenizer.model'):
                sp_model = os.path.join(tokenizer_dir, f)
                break
    if sp_model is None:
        raise FileNotFoundError(f"Could not find a SentencePiece model in tokenizer dir: {tokenizer_dir}")

    tokenizer = SentencePieceTokenizer(sp_model)

    try:
        from nemo.collections.asr.data.audio_to_text import AudioToBPEDataset
        AudioToBPEDataset  # type: ignore
        use_nemo_ds = True
    except Exception:
        use_nemo_ds = False

    from torch.utils.data import DataLoader

    if use_nemo_ds:
        logger.info('Using NeMo AudioToBPEDataset')
        dataset = AudioToBPEDataset(
            manifest_filepath=config.data.train_ds.manifest_filepath,
            tokenizer=tokenizer,
            sample_rate=config.data.train_ds.sample_rate,
            int_values=False,
            max_duration=config.data.train_ds.max_duration,
            min_duration=config.data.train_ds.min_duration,
            trim=config.data.train_ds.trim_silence,
            use_start_end_token=config.data.train_ds.use_start_end_token,
            return_language_id=config.data.train_ds.return_language_id
        )
        collate = getattr(dataset, 'collate_fn', None)
        train_loader = DataLoader(dataset, batch_size=config.data.train_ds.batch_size, shuffle=config.data.train_ds.shuffle, num_workers=config.data.train_ds.num_workers, collate_fn=collate)

        val_dataset = AudioToBPEDataset(
            manifest_filepath=config.data.validation_ds.manifest_filepath,
            tokenizer=tokenizer,
            sample_rate=config.data.validation_ds.sample_rate,
            int_values=False,
            max_duration=config.data.validation_ds.max_duration,
            min_duration=config.data.validation_ds.min_duration,
            trim=config.data.validation_ds.trim_silence,
            use_start_end_token=config.data.validation_ds.use_start_end_token,
            return_language_id=config.data.validation_ds.return_language_id
        )
        val_collate = getattr(val_dataset, 'collate_fn', None)
        val_loader = DataLoader(val_dataset, batch_size=config.data.validation_ds.batch_size, shuffle=config.data.validation_ds.shuffle, num_workers=config.data.validation_ds.num_workers, collate_fn=val_collate)

        # Wrap NeMo DataLoader batches to the six-tuple format expected by the model (signal, signal_len, transcript, transcript_len, sample_ids, language_ids)
        import torch
        class BatchWrapper:
            def __init__(self, loader, model=None):
                self.loader = loader
                self.model = model
            def __iter__(self):
                for batch in self.loader:
                    # If NeMo's AudioToBPEDataset returns (signal, signal_len, labels, labels_len)
                    if isinstance(batch, (list, tuple)) and len(batch) == 4:
                        signal, signal_len, labels, labels_len = batch
                        bs = signal.size(0) if hasattr(signal, 'size') else len(signal)
                        sample_ids = None

                        # Helper: combined language identifier object that behaves like a str for dict lookup
                        # and like an int for int(...) conversion. This avoids modifying library code.
                        class LanguageIdentifier:
                            def __init__(self, key: str, idx: int):
                                self.key = str(key)
                                self.idx = int(idx)
                            def __int__(self):
                                return int(self.idx)
                            def __str__(self):
                                return self.key
                            def __repr__(self):
                                return f"LangId({self.key}:{self.idx})"
                            def __hash__(self):
                                return hash(self.key)
                            def __eq__(self, other):
                                if isinstance(other, LanguageIdentifier):
                                    return self.key == other.key
                                return self.key == other

                        # Provide language_ids as a list of LanguageIdentifier objects so both conv_asr
                        # (which does int(lid)) and joint.ModuleDict (which expects string keys) work.
                        if self.model is not None and hasattr(self.model, 'joint') and hasattr(self.model.joint, 'language_keys'):
                            lang_key = getattr(config, 'custom_config', {}).get('language', 'kok') if hasattr(config, 'custom_config') else 'kok'
                            try:
                                lang_idx = list(self.model.joint.language_keys).index(lang_key)
                            except Exception:
                                lang_idx = 0
                            language_ids = [LanguageIdentifier(lang_key, lang_idx) for _ in range(bs)]
                        else:
                            language_ids = [LanguageIdentifier('kok', 0) for _ in range(bs)]
                        yield (signal, signal_len, labels, labels_len, sample_ids, language_ids)
                    else:
                        # Pass through other batch formats unchanged
                        yield batch
            def __len__(self):
                return len(self.loader)

        wrapped_train = BatchWrapper(train_loader, model=model)
        wrapped_val = BatchWrapper(val_loader, model=model)

        logger.info('Data module setup completed (NeMo dataset)')
        return wrapped_train, wrapped_val

    # Fallback: simple dataset using SentencePiece + soundfile
    logger.info('Falling back to simple dataset (SentencePiece + soundfile)')
    import sentencepiece as spm, soundfile as sf
    sp = spm.SentencePieceProcessor(model_file=sp_model)

    class SimpleASRDataset(torch.utils.data.Dataset):
        def __init__(self, manifest_path):
            self.entries = []
            with open(manifest_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    if line.strip():
                        self.entries.append(json.loads(line))
        def __len__(self):
            return len(self.entries)
        def __getitem__(self, idx):
            e = self.entries[idx]
            audio = e['audio_filepath']
            text = e.get('text','')
            sig, sr = sf.read(audio)
            import numpy as np
            if sr != config.data.train_ds.sample_rate:
                # resample
                try:
                    import resampy
                    sig = resampy.resample(sig, sr, config.data.train_ds.sample_rate)
                except Exception:
                    pass
            sig = np.asarray(sig, dtype='float32')
            if sig.ndim==1:
                sig = np.expand_dims(sig, 0)
            ids = sp.encode(text, out_type=int)
            return {'input_signal': torch.from_numpy(sig), 'input_signal_length': torch.tensor(sig.shape[1], dtype=torch.long), 'labels': torch.tensor(ids, dtype=torch.long), 'labels_length': torch.tensor(len(ids), dtype=torch.long)}

    def collate_fn(batch):
        signals = [b['input_signal'] for b in batch]
        lengths = torch.tensor([s.shape[1] for s in signals], dtype=torch.long)
        maxlen = int(lengths.max())
        padded = torch.zeros((len(signals), 1, maxlen), dtype=torch.float32)
        for i,s in enumerate(signals):
            padded[i, :, :s.shape[1]] = s
        labels = [b['labels'] for b in batch]
        maxlab = max([l.shape[0] for l in labels])
        labpad = torch.zeros((len(labels), maxlab), dtype=torch.long)
        lablen = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)
        for i,l in enumerate(labels):
            labpad[i, :l.shape[0]] = l
        # Return language ids as a Python list of ints to avoid being moved to CUDA as a tensor
        langs = [0] * len(batch)
        return {'input_signal': padded, 'input_signal_length': lengths, 'labels': labpad, 'labels_length': lablen, 'language_ids': langs}

    dataset = SimpleASRDataset(config.data.train_ds.manifest_filepath)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.data.train_ds.batch_size, shuffle=config.data.train_ds.shuffle, collate_fn=collate_fn)
    logger.info('Data module setup completed (fallback simple dataset)')
    # For trainer.fit we return train_loader, val_loader
    # Create a small validation loader from validation manifest if present
    try:
        val_dataset = SimpleASRDataset(config.data.validation_ds.manifest_filepath)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.data.validation_ds.batch_size, shuffle=False, collate_fn=collate_fn)
    except Exception:
        val_loader = None
    return data_loader, val_loader

def setup_trainer(config: DictConfig, output_dir: str):
    """
    Setup PyTorch Lightning trainer

    Args:
        config: Configuration object
        output_dir: Output directory for checkpoints

    Returns:
        PyTorch Lightning trainer
    """
    logger.info("Setting up trainer...")

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback (only add if checkpointing enabled)
    if config.trainer.enable_checkpointing:
        # Respect exp_manager checkpoint params if present
        ckpt_params = getattr(config, 'exp_manager', {}).get('checkpoint_callback_params', {}) if hasattr(config, 'exp_manager') else {}
        save_top_k = ckpt_params.get('save_top_k', 5)
        monitor = ckpt_params.get('monitor', 'val_wer')
        filename = ckpt_params.get('filename', "konkani_asr-{epoch:02d}-{val_wer:.3f}")
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename=filename,
            monitor=monitor,
            mode="min",
            save_top_k=save_top_k,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, "logs"),
        name="konkani_asr_finetune"
    )

    # Create trainer
    trainer_kwargs = dict(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        max_epochs=config.trainer.max_epochs,
        max_steps=config.trainer.max_steps,
        num_nodes=config.trainer.num_nodes,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        enable_checkpointing=config.trainer.enable_checkpointing,
        logger=tb_logger,
        log_every_n_steps=config.trainer.log_every_n_steps,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        callbacks=callbacks,
    )

    # Debugging helper: if FAST_FAIL=1 is set in env, force very small limits to reproduce errors quickly
    if os.environ.get('FAST_FAIL') == '1':
        trainer_kwargs['max_epochs'] = 1
        trainer_kwargs['limit_train_batches'] = 1
        trainer_kwargs['limit_val_batches'] = 1
        logger.info('FAST_FAIL enabled: limiting trainer to 1 batch/epoch for faster repro')

    if hasattr(config.trainer, 'strategy') and config.trainer.strategy not in (None, ''):
        trainer_kwargs['strategy'] = config.trainer.strategy

    # Allow forcing CPU mode for debugging if requested via env var
    if os.environ.get('FORCE_CPU') == '1':
        logger.warning('FORCE_CPU=1 detected: overriding trainer to run on CPU for debugging')
        trainer_kwargs['accelerator'] = 'cpu'
        trainer_kwargs['devices'] = 1

    trainer = pl.Trainer(**trainer_kwargs)

    logger.info("Trainer setup completed")
    return trainer

def fine_tune_model(config: DictConfig, output_dir: str):
    """
    Main fine-tuning function

    Args:
        config: Configuration object
        output_dir: Output directory
    """
    try:
        # Safety: clamp learning rate if config specifies an excessively large value
        try:
            if hasattr(config, 'optim') and getattr(config.optim, 'lr', None) is not None:
                try:
                    cfg_lr = float(config.optim.lr)
                    if cfg_lr > 0.01:
                        logger.warning(f"Large learning rate detected in config.optim.lr={cfg_lr}; clamping to 1e-3 for safety")
                        config.optim.lr = 1e-3
                except Exception:
                    pass
        except Exception:
            pass

        # Setup components
        model = setup_model(config)
        train_loader, val_loader = setup_data_module(config, model=model)
        trainer = setup_trainer(config, output_dir)

        # Create a timestamped experiment folder and write research logging artifacts
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = os.path.join(output_dir, "experiments", timestamp)
            os.makedirs(experiment_dir, exist_ok=True)

            # Save the full config as hyperparameters.json (resolved)
            try:
                hp = OmegaConf.to_container(config, resolve=True)
                with open(os.path.join(experiment_dir, 'hyperparameters.json'), 'w', encoding='utf-8') as fh:
                    json.dump(hp, fh, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to write hyperparameters.json: {e}")

            # Dump model architecture to text file
            try:
                with open(os.path.join(experiment_dir, 'model_architecture.txt'), 'w', encoding='utf-8') as fh:
                    fh.write(str(model))
            except Exception as e:
                logger.warning(f"Failed to write model_architecture.txt: {e}")

            # CSV epoch metrics file
            csv_path = os.path.join(experiment_dir, 'epoch_metrics.csv')
            if not os.path.exists(csv_path):
                try:
                    with open(csv_path, 'w', newline='') as fh:
                        writer = csv.writer(fh)
                        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_wer', 'val_char_dist', 'lr', 'time_elapsed'])
                except Exception as e:
                    logger.warning(f"Failed to create epoch CSV file: {e}")


            # Instantiate callbacks and attach to trainer
            try:
                csv_logger = ResearchCSVLogger(csv_path)
                sample_logger = SampleLoggerDebug(getattr(config.data.validation_ds, 'manifest_filepath', ''), experiment_dir)
                trainer.callbacks.append(csv_logger)
                trainer.callbacks.append(sample_logger)

                # Force optimizer LR to config value for reproducibility
                try:
                    lr_val = float(getattr(config, 'optim', {}).get('lr', 0.0))
                    force_lr_cb = ForceLRCallback(lr_val)
                    trainer.callbacks.append(force_lr_cb)
                    logger.info(f"ForceLRCallback added to set lr={lr_val}")
                except Exception as e:
                    logger.warning(f"Failed to add ForceLRCallback: {e}")

                logger.info(f"Research logging initialized in {experiment_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize research logger callbacks: {e}")

        except Exception as e:
            logger.warning(f"Failed to create experiment folder or logging artifacts: {e}")

        # Setup experiment manager
        if hasattr(config, 'exp_manager'):
            # exp_manager expects a more fully-specified config; try to run it but do not fail training if it errors
            try:
                config.exp_manager.exp_dir = output_dir
                from nemo.utils import exp_manager as em
                em.exp_manager(config.exp_manager, trainer)
            except Exception as e:
                logger.warning(f"exp_manager invocation failed: {e}")

        # Start training
        logger.info("Starting fine-tuning...")
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Print final training metrics if available
        try:
            metrics = trainer.callback_metrics
            logger.info(f"Final callback metrics: {metrics}")
            # Try to extract training loss
            final_train_loss = None
            if 'train_loss' in metrics:
                final_train_loss = metrics['train_loss']
                logger.info(f"Final training loss: {final_train_loss}")

            # Post-process CSV to ensure train_loss is recorded (best-effort)
            try:
                # locate experiment dir from attached CSV logger
                exp_dir = None
                for c in trainer.callbacks:
                    if c.__class__.__name__ == 'ResearchCSVLogger':
                        exp_dir = os.path.dirname(c.filepath)
                        csv_path = c.filepath
                        break
                if exp_dir and os.path.exists(csv_path) and final_train_loss is not None:
                    import csv as _csv, io as _io
                    # Read CSV
                    with open(csv_path, 'r', newline='') as fh:
                        rows = list(_csv.reader(fh))
                    if len(rows) >= 2:
                        header = rows[0]
                        last_row = rows[-1]
                        # header expected: ['epoch','train_loss','val_loss','val_wer','lr','time_elapsed']
                        try:
                            train_idx = header.index('train_loss')
                            if last_row[train_idx] == '' or last_row[train_idx] is None:
                                last_row[train_idx] = str(float(final_train_loss))
                                rows[-1] = last_row
                                # Write back
                                with open(csv_path, 'w', newline='') as fh:
                                    writer = _csv.writer(fh)
                                    writer.writerows(rows)
                                logger.info(f"Backfilled train_loss={final_train_loss} into {csv_path}")
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Failed to post-process epoch CSV: {e}")

        except Exception as e:
            logger.warning(f"Could not read final metrics: {e}")


        logger.info("Fine-tuning completed successfully!")

        # Final test run using best checkpoint (if available) and save results to experiment folder
        try:
            # Locate experiment directory from attached CSV logger (if created)
            experiment_dir = None
            for c in trainer.callbacks:
                if c.__class__.__name__ == 'ResearchCSVLogger':
                    experiment_dir = os.path.dirname(c.filepath)
                    break
            if experiment_dir is None:
                experiment_dir = os.path.join(output_dir, 'experiments', datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                os.makedirs(experiment_dir, exist_ok=True)

            # Find best checkpoint
            best_ckpt = None
            for c in trainer.callbacks:
                if isinstance(c, ModelCheckpoint) and getattr(c, 'best_model_path', ''):
                    best_ckpt = c.best_model_path
                    break
            if not best_ckpt and hasattr(trainer, 'checkpoint_callback'):
                cb = getattr(trainer, 'checkpoint_callback')
                if cb is not None and getattr(cb, 'best_model_path', ''):
                    best_ckpt = cb.best_model_path

            if best_ckpt:
                logger.info(f"Running final evaluation using best checkpoint: {best_ckpt}")
                try:
                    # Try to perform a trustful load from the checkpoint in several ways.
                    # 1) Prefer class-level load_from_checkpoint (trusted deserialization)
                    # 2) Try to apply full state_dict with prefix-stripping (module., model.)
                    # 3) Fallback to filtered matching by exact key/shape
                    import torch
                    try:
                        ModelClass = model.__class__
                        logger.info('Attempting to load checkpoint via ModelClass.load_from_checkpoint(...)')
                        loaded_model = ModelClass.load_from_checkpoint(best_ckpt, map_location='cpu')
                        model = loaded_model
                        logger.info('Successfully loaded model via load_from_checkpoint')
                    except Exception as e_load:
                        logger.warning(f"load_from_checkpoint failed: {e_load}; attempting state_dict-based mapping")

                        try:
                            ckpt = torch.load(best_ckpt, map_location='cpu')
                        except Exception as e0:
                            # Retry with trustful load (weights_only=False) per user request
                            logger.warning('Safe torch.load failed; retrying with weights_only=False (trust required)')
                            ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=False)

                        state = ckpt.get('state_dict', ckpt)

                        # Helper to try stripping common prefixes
                        def _strip_prefixes(sd):
                            new = {}
                            for k, v in sd.items():
                                kk = k
                                for p in ('model.', 'module.'):
                                    if kk.startswith(p):
                                        kk = kk[len(p):]
                                new[kk] = v
                            return new

                        base_sd = model.state_dict()

                        # Try direct assignment first
                        try:
                            model.load_state_dict(state, strict=False)
                            logger.info('Loaded checkpoint state_dict directly into model (strict=False)')
                        except Exception:
                            # Try prefix-stripped keys
                            stripped = _strip_prefixes(state)
                            matched = 0
                            for k, v in stripped.items():
                                if k in base_sd and list(v.shape) == list(base_sd[k].shape):
                                    matched += 1
                            total = len(base_sd)
                            if matched >= max(1, int(0.6 * total)):
                                # If a reasonable fraction matches, apply
                                model.load_state_dict(stripped, strict=False)
                                logger.info(f'Loaded checkpoint after prefix-stripping; matched {matched}/{total} params')
                            else:
                                # Last resort: filtered matching keyed by exact matches (old behavior)
                                filtered = {}
                                matched = skipped = 0
                                for k, v in state.items():
                                    if k in base_sd and list(v.shape) == list(base_sd[k].shape):
                                        filtered[k] = v
                                        matched += 1
                                    else:
                                        skipped += 1
                                logger.info(f'Applying fallback filtered mapping: matched {matched} params, skipped {skipped} params')
                                model.load_state_dict(filtered, strict=False)
                except Exception as e:
                    logger.warning(f"Failed to load best checkpoint: {e}")

                # Force CTC decoding for final evaluation (avoid RNNT decoder path if untrained)
                try:
                    if hasattr(model, 'change_decoding_strategy'):
                        print("Forcing CTC decoding for final test phase...", flush=True)
                        logger.info('Forcing CTC decoding for final test phase...')
                        try:
                            model.change_decoding_strategy(decoder_type='ctc')
                        except Exception:
                            pass
                    try:
                        model.cur_decoder = 'ctc'
                    except Exception:
                        pass
                except Exception:
                    pass

                # Evaluate on validation manifest (create per-sample and summary results)
                manifest = getattr(config.data.validation_ds, 'manifest_filepath', None)
                if manifest and os.path.exists(manifest):
                    per_sample = []
                    try:
                        with open(manifest, 'r', encoding='utf-8') as fh:
                            for line in fh:
                                if not line.strip():
                                    continue
                                obj = json.loads(line)
                                audio = obj.get('audio_filepath')
                                ref = obj.get('text', '')
                                pred = ''
                                wer_val = None
                                try:
                                    if hasattr(model, 'transcribe') and audio:
                                        # Robust per-sample transcription: try multiple call forms and kwargs and prefer non-empty outputs
                                        prev_dec = getattr(model, 'cur_decoder', None)
                                        try:
                                            try:
                                                model.cur_decoder = 'ctc'
                                            except Exception:
                                                pass

                                            def _normalize_pred(p):
                                                import ast
                                                if isinstance(p, str):
                                                    try:
                                                        parsed = ast.literal_eval(p)
                                                        return _normalize_pred(parsed)
                                                    except Exception:
                                                        return p
                                                if isinstance(p, (list, tuple)) and len(p) > 0:
                                                    first = p[0]
                                                    if isinstance(first, (list, tuple)) and len(first) > 0:
                                                        return _normalize_pred(first[0])
                                                    return _normalize_pred(first)
                                                return str(p)

                                            def _map_language_id(mod, lid_str):
                                                mapped = None
                                                try:
                                                    if hasattr(mod, 'joint') and hasattr(mod.joint, 'language_keys'):
                                                        mapped = list(mod.joint.language_keys).index(lid_str)
                                                except Exception:
                                                    mapped = None
                                                if mapped is None:
                                                    try:
                                                        if hasattr(mod, 'cfg') and 'language_keys' in getattr(mod.cfg, 'joint', {}):
                                                            mapped = list(mod.cfg.joint.language_keys).index(lid_str)
                                                    except Exception:
                                                        mapped = None
                                                return mapped

                                            got = None
                                            candidate_kwargs = [ {'language_id': 'kok', 'batch_size': 1, 'logprobs': False}, {'batch_size': 1, 'logprobs': False}, {'logprobs': False}, {} ]
                                            for kw in candidate_kwargs:
                                                try:
                                                    kwargs = dict(kw)
                                                    if 'language_id' in kwargs and isinstance(kwargs['language_id'], str):
                                                        mapped = _map_language_id(model, kwargs['language_id'])
                                                        if mapped is not None:
                                                            kwargs['language_id'] = mapped
                                                    try:
                                                        out = model.transcribe([audio], **kwargs)
                                                    except TypeError:
                                                        try:
                                                            out = model.transcribe(paths2audio_files=[audio], **kwargs)
                                                        except TypeError:
                                                            out = model.transcribe(audio, **kwargs)
                                                    got = _normalize_pred(out)
                                                    if isinstance(got, str) and got != '':
                                                        break
                                                except Exception:
                                                    got = None
                                                    continue
                                            pred = got if got is not None else ''
                                        finally:
                                            try:
                                                if prev_dec is None:
                                                    if hasattr(model, 'cur_decoder'):
                                                        delattr(model, 'cur_decoder')
                                                else:
                                                    model.cur_decoder = prev_dec
                                            except Exception:
                                                pass
                                    else:
                                        pred = ''
                                    wer_val = _compute_wer(ref, pred) if ref else None
                                except Exception as e:
                                    logger.warning(f"Failed to transcribe {audio}: {e}")
                                per_sample.append({'audio': audio, 'reference': ref, 'prediction': pred, 'wer': wer_val})
                        total_wer = sum([p['wer'] for p in per_sample if p['wer'] is not None]) / max(1, len([p for p in per_sample if p['wer'] is not None]))
                        out = {'best_checkpoint': best_ckpt, 'summary': {'total_samples': len(per_sample), 'mean_wer': total_wer}, 'per_sample': per_sample}
                        outpath = os.path.join(experiment_dir, 'final_test_results.json')
                        with open(outpath, 'w', encoding='utf-8') as fh:
                            json.dump(out, fh, indent=2, ensure_ascii=False)
                        logger.info(f"Wrote final test results to {outpath}")

                        # Optional: export a .nemo artifact from the best checkpoint if explicitly requested
                        try:
                            if os.environ.get('SAVE_NEMO_ARTIFACT') == '1':
                                # Check free space (bytes available on the experiment dir mount)
                                statvfs = os.statvfs(experiment_dir)
                                free_bytes = statvfs.f_bavail * statvfs.f_frsize
                                # Require at least 10GB free to safely write the .nemo
                                if free_bytes < (10 * 1024 ** 3):
                                    logger.warning(f"Insufficient free space ({free_bytes / 1024 ** 3:.2f} GB) to save .nemo artifact; skipping .nemo export")
                                else:
                                    nemo_name = f"{os.path.basename(best_ckpt) if best_ckpt else 'model'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.nemo"
                                    nemo_path = os.path.join(experiment_dir, nemo_name)
                                    try:
                                        # Prefer model.save_to() if available
                                        if hasattr(model, 'save_to'):
                                            model.save_to(nemo_path)
                                        elif hasattr(model, 'save_to_nemo'):
                                            model.save_to_nemo(nemo_path)
                                        else:
                                            # Fallback: try NeMo serializer
                                            try:
                                                import nemo.collections.asr as nemo_asr
                                                model.save_to(nemo_path)
                                            except Exception as e:
                                                raise RuntimeError(f"No known .nemo save method available: {e}")
                                        logger.info(f"Saved .nemo artifact to {nemo_path}")
                                    except Exception as e:
                                        logger.warning(f"Failed to save .nemo artifact: {e}")
                        except Exception as e:
                            logger.warning(f".nemo export step failed: {e}")

                    except Exception as e:
                        logger.warning(f"Final evaluation failed: {e}")
                else:
                    logger.warning('Validation manifest not found; skipping final evaluation')
            else:
                logger.warning('No best checkpoint found; skipping final evaluation')
        except Exception as e:
            logger.warning(f"Final test run failed: {e}")

    except NeMoBaseException as e:
        logger.error(f"NeMo error during fine-tuning: {e}")
        raise
    except Exception as e:
        # Log full traceback to aid debugging (was only logging the exception string which may be None)
        logger.exception("Unexpected error during fine-tuning")
        raise

def main():
    parser = argparse.ArgumentParser(description="Fine-tune IndicConformer for Konkani ASR")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    parser.add_argument("--output_dir", default="results", help="Output directory for results")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Load configuration
        config = load_config(args.config)

        # Override output directory in config
        config.exp_manager.exp_dir = args.output_dir

        # Start fine-tuning
        fine_tune_model(config, args.output_dir)

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()