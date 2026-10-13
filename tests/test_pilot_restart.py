"""Small-pilot restart and full-backup behavior; no network or accelerator."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

import torch
from dsa.data.pilot_prepare import prepare_pilot
from dsa.training.pilot import PilotConfig, train_pilot
from scripts.ensure_pilot_arm import ensure_arm, restore_backup
from scripts.package_pilot_review import package_review
from scripts.screen_recurrence import screen

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class PilotRestartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'source'
        source.mkdir()
        for split in ('train', 'valid', 'test'):
            (source / f'gsm_{split}.json').write_text(json.dumps([
                {'question': f'{split}: add two and three?', 'steps': ['<<2+3=5>>'], 'answer': '5'}]))
        self.prepared = self.root / 'prepared'
        prepare_pilot(storage_dir=None, source_dir=source, output_dir=self.prepared, loops=2, tiny=True, train_limit=0)
        self.legacy = self.root / 'drive/runs/example'
        self.working = self.root / 'vm/example'
        self.backups = self.root / 'drive/bundles/example/checkpoints'
        with contextlib.redirect_stdout(io.StringIO()):
            train_pilot(PilotConfig(self.prepared, self.legacy / 'static', tiny=True, loops=2,
                                   steps=2, save_every=2, device='cpu', write_mode='static'))

    def ensure(self, arm='static', **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return ensure_arm(self.prepared, self.working / arm, self.legacy / arm,
                self.backups, self.root / f'reports/{arm}.json', arm=arm, steps=2, device='cpu',
                template_dir=self.legacy / 'static', **kwargs)

    def test_completed_static_reused_and_metadata_only_learned_retrained_matched(self):
        # Reproduce the user's exact situation: stale learned metadata, no weights.
        learned = self.legacy / 'learned'
        learned.mkdir()
        (learned / 'latest.json').write_text('{"step":75,"path":"checkpoint-00000075"}')
        with mock.patch('scripts.ensure_pilot_arm.pilot.train_pilot', side_effect=AssertionError('must reuse static')):
            static = self.ensure()
        result = self.ensure('learned', expected_initial_sha=static['initial_state_sha256'])
        self.assertEqual((static['action'], result['action']), ('reused', 'restarted'))
        self.assertEqual(result['step'], 2)
        self.assertEqual(result['initial_state_sha256'], static['initial_state_sha256'])
        self.assertEqual(json.loads((learned / 'latest.json').read_text())['step'], 75)
        self.assertTrue(Path(result['drive_full_backup']).is_file())

    def test_missing_vm_and_historical_files_restore_from_full_drive_archive(self):
        first = self.ensure()
        shutil.rmtree(self.working)
        shutil.rmtree(self.legacy)
        with mock.patch('scripts.ensure_pilot_arm.pilot.train_pilot', side_effect=AssertionError('archive should restore')):
            second = self.ensure()
        self.assertEqual(second['action'], 'reused')
        self.assertEqual(first['initial_state_sha256'], second['initial_state_sha256'])
        self.assertTrue((self.working / 'static/checkpoint-00000002/model/anchor.pt').is_file())

    def test_complete_marker_without_model_bytes_restarts(self):
        for path in (self.legacy / 'static/checkpoint-00000002/model/backbone').glob('*.safetensors'):
            path.unlink()
        result = self.ensure()
        self.assertEqual(result['action'], 'restarted')
        self.assertEqual(result['step'], 2)

    def test_partial_working_run_restarts_instead_of_resuming(self):
        with contextlib.redirect_stdout(io.StringIO()):
            train_pilot(PilotConfig(self.prepared, self.working / 'learned', tiny=True, loops=2,
                                   steps=1, device='cpu', write_mode='learned'))
        result = self.ensure('learned')
        self.assertEqual(result['action'], 'restarted')
        metrics = [json.loads(line) for line in (self.working / 'learned/metrics.jsonl').read_text().splitlines()]
        self.assertEqual([row['step'] for row in metrics], [1, 2])
        self.assertEqual(len(list(self.working.glob('learned-previous-*'))), 1)

    def test_initialization_mismatch_stops_before_optimizer_updates(self):
        with mock.patch('torch.optim.AdamW.step', side_effect=AssertionError('no update allowed')) as step:
            with self.assertRaisesRegex(ValueError, 'Fresh initialization differs'):
                self.ensure('learned', expected_initial_sha='incorrect')
        step.assert_not_called()

    def test_drive_copy_failure_keeps_independent_full_backup(self):
        with mock.patch('scripts.ensure_pilot_arm.shutil.copy2', side_effect=OSError('Drive unavailable')):
            result = self.ensure('learned')
        self.assertEqual(result['backup_status'], 'failed')
        local = Path(result['local_full_backup'])
        self.assertTrue(local.is_file())
        restored = self.root / 'independent-restore'
        restore_backup(local, restored)
        self.assertTrue((restored / 'checkpoint-00000002/training_state.pt').is_file())
        self.assertTrue((restored / 'checkpoint-00000002/model/backbone/model.safetensors').is_file())

    def test_force_restart_ignores_even_good_completed_checkpoints(self):
        self.assertEqual(self.ensure(force_restart=True)['action'], 'restarted')

    def test_nonfinite_initialization_screen_retains_failure_report(self):
        fake={'loop_budget':1,'raw_residuals':[{'finite':False}], 'losses':{'answer':{
            'parameter_gradients':{'global_norm_before_clipping':None,'nonfinite_parameter_gradients':1}}}}
        out=self.root/'screen'
        with mock.patch('scripts.screen_recurrence.gradient_probe',return_value=fake), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'screen failed'):
                screen(self.prepared,out,device='cpu',depths=[1],conditions={
                    'static':{'write_mode':'static','recurrence_mode':'standard'}})
        self.assertFalse(json.loads((out/'report.json').read_text())['finite'])

    def test_completed_screen_reuses_results_without_reloading_model(self):
        out=self.root/'screen'
        with contextlib.redirect_stdout(io.StringIO()):
            result=screen(self.prepared,out,device='cpu',depths=[1,2])
            with mock.patch('scripts.screen_recurrence.pilot._initial_model',side_effect=AssertionError('cached screen')):
                again=screen(self.prepared,out,device='cpu',depths=[1,2])
        self.assertEqual(result,again)
        self.assertEqual(result['optimizer_updates'],0)

    def test_export_reads_local_working_journal_with_drive_reports(self):
        self.ensure()
        project = self.root / 'drive'
        with contextlib.redirect_stdout(io.StringIO()):
            package_review(project, 'example', 'review', self.prepared, run_root=self.working)
        with ZipFile(project / 'bundles/example-review.zip') as archive:
            self.assertIn('runs/static/metrics.jsonl', archive.namelist())
            report = json.loads(archive.read('review/scale_readiness.json'))
            self.assertEqual(report['arms']['static']['training_steps_logged'], 2)
            self.assertTrue(report['missing_required_artifacts'])

    def test_notebook_auto_restart_runs_through_diagnostics_and_both_exports(self):
        import hashlib, os, runpy, sys, types
        project = self.root / 'drive'
        (project / 'prepared').mkdir()
        shutil.move(self.prepared, project / 'prepared/cache')
        # Manifests identify prepared contents by checksum, independent of path.
        self.prepared = project / 'prepared/cache'
        notebook = json.loads((ROOT / 'notebooks/DSA_Colab.ipynb').read_text())
        sources = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        config = sources['config'].replace('EXISTING_RUN_NAME = f"gpt2-k8-flt-v1-{REPO_COMMIT[:12]}"', 'EXISTING_RUN_NAME = "example"')
        for old, new in [
            ('7, 32, 8, 4, 200', '7, 1, 2, 4, 2'), ('DEVICE = "cuda"', 'DEVICE = "cpu"'),
            ('VALID_LIMIT = 128', 'VALID_LIMIT = 1'), ('MAX_NEW_TOKENS = 64', 'MAX_NEW_TOKENS = 2'),
            ('4, 8, 8', '1, 1, 1'), ('[1, 2, 4, 8, 16]', '[1, 2]'),
            ('COT_MAX_NEW_TOKENS = 192', 'COT_MAX_NEW_TOKENS = 2'),
            ('SCREEN_LIMIT, SCREEN_DEPTHS = 1, [1, 8, 16]', 'SCREEN_LIMIT, SCREEN_DEPTHS = 1, [1, 2]'),
            ('Path("/content/dsa-pilot-working")', f'Path({str(self.root / "vm")!r})'),
            ('Path("/content/dsa-prepared")', f'Path({str(self.root / "inputs")!r})'),
        ]: config = config.replace(old, new)
        namespace = dict(Path=Path, json=json, hashlib=hashlib, shutil=shutil, os=os,
                         PROJECT_DIR=project, REPO_DIR=ROOT, REPO_COMMIT='fixture',
                         PYTHON=sys.executable, GPU_READY=True, SMOKE_READY=True)
        downloads = []
        colab = types.ModuleType('google.colab')
        colab.files = types.SimpleNamespace(download=lambda path: downloads.append(Path(path)))
        def run_project(script, *args):
            with mock.patch.object(sys, 'argv', [script, *map(str, args)]):
                runpy.run_path(str(ROOT / script), run_name='__main__')
        with mock.patch.dict(sys.modules, {'google.colab': colab}), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(config, 'config', 'exec'), namespace)
            namespace['run_project'] = run_project
            for name in ('data', 'initialization-screen', 'training', 'evaluation', 'pilot-diagnostics', 'pilot-controls',
                         'final-evaluation', 'summary', 'pilot-export'):
                source = sources[name].replace('Path("/content")', f'Path({str(self.root)!r})')
                exec(compile(source, name, 'exec'), namespace)
        self.assertEqual(namespace['READY_ARMS'], ['static', 'learned', 'flt_static', 'flt_learned'])
        self.assertEqual(namespace['arm_checkpoints']['static']['action'], 'reused')
        self.assertEqual(namespace['arm_checkpoints']['learned']['action'], 'restarted')
        self.assertEqual(len(downloads), 5)
        for path in downloads[:4]:
            with ZipFile(path) as backup:
                self.assertIn('checkpoint-00000002/training_state.pt', backup.namelist())
        with ZipFile(downloads[-1]) as review:
            manifest = json.loads(review.read('bundle_manifest.json'))
            readiness = json.loads(review.read('review/scale_readiness.json'))
            self.assertTrue(manifest['evidence_complete'])
            self.assertTrue(readiness['matched_controls']['learned']['matched'])
            self.assertTrue(readiness['matched_controls']['flt_learned']['matched'])
            self.assertEqual(len(readiness['factorial_comparisons']), 4)
            self.assertTrue(readiness['initialization_screen']['finite'])
            profile=json.loads(review.read('review/flt_learned/diagnostics/forward-profiles/sample0000.json'))
            self.assertEqual(len(profile['raw_residuals']), 10)
            probe=json.loads(review.read('review/flt_learned/diagnostics/gradients/sample0000-k2-joint.json'))
            self.assertTrue(all(x['connected_to_loss'] for x in probe['losses']['answer']['states']))
            self.assertFalse(readiness['missing_required_artifacts'])


if __name__ == '__main__': unittest.main()
