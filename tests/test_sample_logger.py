import os
import json
import tempfile
from scripts.fine_tune import SampleLoggerCallback


class FakeTrainer:
    def __init__(self):
        self.current_epoch = 0
        self.global_step = 10
        self.callback_metrics = {}


class FakeModule:
    def __init__(self, response):
        self._response = response
        self._cfg = {'name': 'FakeModel'}

    def transcribe(self, *args, **kwargs):
        # Return the preconfigured response irrespective of input
        return self._response

    def __class__(self):
        return self


def write_manifest(path, text='पाव'):
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps({'audio_filepath': 'dummy.wav', 'text': text}) + '\n')


def test_sample_logger_normalizes_simple_string(tmp_path):
    mpath = tmp_path / 'manifest.jsonl'
    outdir = tmp_path / 'out'
    write_manifest(mpath)

    cb = SampleLoggerCallback(str(mpath), str(outdir), max_samples=1)
    trainer = FakeTrainer()
    # Module that returns nested list shapes similar to some transcribe outputs
    responses = [['पाव']]
    mod = FakeModule(responses)

    cb.on_validation_epoch_end(trainer, mod)

    out_file = outdir / 'samples_epoch_00.json'
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding='utf-8'))
    assert data['samples'][0]['pred'] in ['पाव', 'पाव']
    assert data['samples'][0]['deva_ok'] is True


def test_sample_logger_handles_various_structures(tmp_path):
    mpath = tmp_path / 'manifest.jsonl'
    outdir = tmp_path / 'out'
    write_manifest(mpath, text='पाव')

    cb = SampleLoggerCallback(str(mpath), str(outdir), max_samples=1)
    trainer = FakeTrainer()
    # Different possible transcribe outputs
    for resp in ['पाव', ['पाव'], [['पाव']], [[['पाव']]]]:
        mod = FakeModule(resp)
        cb.on_validation_epoch_end(trainer, mod)
        out_file = outdir / 'samples_epoch_00.json'
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding='utf-8'))
        assert data['samples'][0]['pred'] == 'पाव'
        assert data['samples'][0]['deva_ok'] is True
